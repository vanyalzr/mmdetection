import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        model.dim = 0

        # Copyright (c) Open-MMLab. All rights reserved.
        import torch
        from torch.nn.parallel._functions import Scatter as OrigScatter
        from mmcv.parallel.data_container import DataContainer

        def scatter_(inputs, target_gpus, dim=0):
            """Scatter inputs to target gpus.

            The only difference from original :func:`scatter` is to add support for
            :type:`~mmcv.parallel.DataContainer`.
            """

            def scatter_map(obj):
                if isinstance(obj, torch.Tensor):
                    return OrigScatter.apply(target_gpus, None, dim, obj)
                if isinstance(obj, DataContainer):
                    return obj.data
                if isinstance(obj, tuple) and len(obj) > 0:
                    return list(zip(*map(scatter_map, obj)))
                if isinstance(obj, list) and len(obj) > 0:
                    out = list(map(list, zip(*map(scatter_map, obj))))
                    return out
                if isinstance(obj, dict) and len(obj) > 0:
                    out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
                    return out
                return [obj for targets in target_gpus]

            # After scatter_map is called, a scatter_map cell will exist. This cell
            # has a reference to the actual function scatter_map, which has references
            # to a closure that has a reference to the scatter_map cell (because the
            # fn is recursive). To avoid this reference cycle, we set the function to
            # None, clearing the cell
            try:
                return scatter_map(inputs)
            finally:
                scatter_map = None

        def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
            """Scatter with support for kwargs dictionary"""
            inputs = scatter_(inputs, target_gpus, dim) if inputs else []
            kwargs = scatter_(kwargs, target_gpus, dim) if kwargs else []
            if len(inputs) < len(kwargs):
                inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
            elif len(kwargs) < len(inputs):
                kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
            inputs = tuple(inputs)
            kwargs = tuple(kwargs)
            return inputs, kwargs

        def scatter(self, inputs, kwargs, device_ids):
            return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

        def train_step(self, *inputs, **kwargs):
            inputs, kwargs = self.scatter(inputs, kwargs, [0])
            return self.module.train_step(*inputs[0], **kwargs[0])

        model.train_step = train_step.__get__(model)
        model.scatter = scatter.__get__(model)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
