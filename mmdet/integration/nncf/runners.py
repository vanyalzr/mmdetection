import warnings
import time

import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner.utils import get_host_info
from mmdet.core.evaluation.eval_hooks import EvalHook


class AccuracyAwareRunner(EpochBasedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data_loaders, workflow, max_epochs=None, compression_ctrl=None, **kwargs):
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        self.data_loaders = data_loaders

        compression_ctrl.run_accuracy_aware_training(kwargs['nncf_config'])

    def train_fn(self, config, compression_ctrl, model, epoch, optimizers, lr_schedulers):
        self.train(self.data_loaders[0])

    def validation_fn(self, model, config):
        for hook in self._hooks:
            if isinstance(hook, EvalHook):
                return hook.eval_result

    def on_training_end_fn(self):
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
