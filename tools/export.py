# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import os.path as osp
import sys
from subprocess import run, check_call, CalledProcessError, DEVNULL

import mmcv
import numpy as np
import onnx
import torch
from onnx.optimizer import optimize
from torch.onnx.symbolic_helper import _onnx_stable_opsets as available_opsets

from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmdet.models import detectors
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models.roi_heads import SingleRoIExtractor
from mmdet.utils.deployment.symbolic import register_extra_symbolics
from mmdet.utils.deployment.ssd_export_helpers import *
from mmdet.utils.deployment.tracer_stubs import AnchorsGridGeneratorStub, ROIFeatureExtractorStub


def export_to_onnx(model,
                   data,
                   export_name,
                   verbose=False,
                   strip_doc_string=False,
                   opset=10,
                   alt_ssd_export=False):
    register_extra_symbolics(opset)

    kwargs = {}
    if torch.__version__ >= '1.5':
        kwargs['enable_onnx_checker'] = False

    if alt_ssd_export:
        assert isinstance(model, detectors.SingleStageDetector)
        
        model.onnx_export = onnx_export.__get__(model)
        model.forward = forward.__get__(model)
        model.forward_export = forward_export_detector.__get__(model)
        model.bbox_head.export_forward = export_forward_ssd_head.__get__(model.bbox_head)
        model.bbox_head._prepare_cls_scores_bbox_preds = prepare_cls_scores_bbox_preds_ssd_head.__get__(model.bbox_head)
        
        model.onnx_export(img=data['img'][0],
                          img_metas=data['img_metas'][0],
                          export_name=export_name,
                          verbose=verbose,
                          opset_version=opset,
                          strip_doc_string=strip_doc_string,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          input_names=['image'],
                          output_names=['detection_out'],
                          keep_initializers_as_inputs=True,
                          **kwargs)
    else:
        output_names = ['boxes', 'labels']
        dynamic_axes = {
            'image': {2: 'height', 3: 'width'},
            'boxes': {0: 'objects_num'},
            'labels': {0: 'objects_num'}
        }
        if hasattr(model, 'roi_head'):
            if model.roi_head.with_mask:
                output_names.append('masks')
                dynamic_axes['masks'] = {0: 'objects_num'}

        with torch.no_grad():
            model.export(
                **data,
                export_name=export_name,
                verbose=verbose,
                opset_version=opset,
                strip_doc_string=strip_doc_string,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                input_names=['image'],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=True,
                **kwargs
                )


def check_onnx_model(export_name):
    try:
        onnx.checker.check_model(export_name)
        print('ONNX check passed.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed.')
        print(ex)


def add_node_names(export_name):
    model = onnx.load(export_name)
    for n in model.graph.node:
        if not n.name:
            n.name = '_'.join([i for i in n.output])
    onnx.save(model, export_name)


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)
    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField('name')
    onnx.save(onnx_model, onnx_model_path)
    output_names = ','.join(output_names)
    

    assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
    normalize = [v for v in cfg.data.test.pipeline[1]['transforms']
                 if v['type'] == 'Normalize'][0]

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{output_dir_path}" ' \
                   f'--output="{output_names}"' 
    if input_shape is not None:
        command_line += f' --input_shape="{input_shape}"'
    if normalize['to_rgb']:
        command_line += ' --reverse_input_channels'

    try:
        run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError as ex:
        print('OpenVINO Model Optimizer not found, please source '
              'openvino/bin/setupvars.sh before running this script.')
        return

    print(command_line)
    run(command_line, shell=True, check=True)


def stub_anchor_generator(model, anchor_head_name):
    anchor_head = getattr(model, anchor_head_name, None)
    if anchor_head is not None and isinstance(anchor_head, AnchorHead):
        anchor_generator = anchor_head.anchor_generator
        anchor_generator.is_stubbed = True
        num_levels = anchor_generator.num_levels
        strides = anchor_generator.strides

        single_level_grid_anchors_generators = []
        for i in range(num_levels):
            single_level_grid_anchors_generator = AnchorsGridGeneratorStub(anchor_generator.single_level_grid_anchors)
            # Save base anchors as operation parameter. It's used at ONNX export time during symbolic call.
            single_level_grid_anchors_generator.params['base_anchors'] = anchor_generator.base_anchors[i].cpu().numpy()
            single_level_grid_anchors_generators.append(single_level_grid_anchors_generator)

        def grid_anchors(self, featmap_sizes, device='cuda'):

            assert num_levels == len(featmap_sizes)
            multi_level_anchors = []
            for i in range(num_levels):
                anchors = single_level_grid_anchors_generators[i](
                    anchor_generator.base_anchors[i].to(device),
                    featmap_sizes[i],
                    stride=strides[i],
                    device=device)
                multi_level_anchors.append(anchors)
            return multi_level_anchors

        anchor_generator.grid_anchors = grid_anchors.__get__(anchor_generator)


def stub_roi_feature_extractor(model, extractor_name):
    if hasattr(model, extractor_name):
        extractor = getattr(model, extractor_name)
        if isinstance(extractor, SingleRoIExtractor):
            setattr(model, extractor_name, ROIFeatureExtractorStub(extractor))
        elif isinstance(extractor, torch.nn.ModuleList):
            for i in range(len(extractor)):
                if isinstance(extractor[i], SingleRoIExtractor):
                    extractor[i] = ROIFeatureExtractorStub(extractor[i])


def get_fake_input(cfg, orig_img_shape=(128, 128, 3), device='cuda'):
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    image = 'data/coco/val2017/000000397133.jpg'
    data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8)) #dict(img=image)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data


def optimize_onnx_graph(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = optimize(onnx_model, ['extract_constant_to_initializer',
                                       'eliminate_unused_initializer'])

    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(onnx_model, onnx_model_path)


def main(args):
    assert args.opset in available_opsets
    assert args.opset > 9

    torch.set_default_tensor_type(torch.FloatTensor)
    model = init_detector(args.config, args.checkpoint, device='cpu')
    model.eval()
    model.cuda()
    device = next(model.parameters()).device
    cfg = model.cfg
    fake_data = get_fake_input(cfg, device=device)

    if args.target == 'openvino' and not args.alt_ssd_export:
        stub_anchor_generator(model, 'rpn_head')
        stub_anchor_generator(model, 'bbox_head')
        if hasattr(model, 'roi_head'):
            stub_roi_feature_extractor(model.roi_head, 'bbox_roi_extractor')
            stub_roi_feature_extractor(model.roi_head, 'mask_roi_extractor')

    mmcv.mkdir_or_exist(osp.abspath(args.output_dir))
    onnx_model_path = osp.join(args.output_dir,
                               osp.splitext(osp.basename(args.config))[0] + '.onnx')
    
    with torch.no_grad():
        export_to_onnx(model, fake_data, export_name=onnx_model_path, opset=args.opset,
                       alt_ssd_export=getattr(args, 'alt_ssd_export', False))
        add_node_names(onnx_model_path)
        print(f'ONNX model has been saved to "{onnx_model_path}"')

    optimize_onnx_graph(onnx_model_path)

    if args.target == 'openvino':
        input_shape = list(fake_data['img'][0].shape)
        if args.input_shape:
            input_shape = [1, 3, *args.input_shape]
        export_to_openvino(cfg, onnx_model_path, args.output_dir, input_shape)
    else:
        pass
        # Model check raises a Segmentation Fault in the latest (1.6.0, 1.7.0) versions of onnx package.
        # Even for a valid graph.
        # check_onnx_model(onnx_model_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output_dir', help='path to directory to save exported models in')
    parser.add_argument('--opset', type=int, default=10, help='ONNX opset')

    subparsers = parser.add_subparsers(title='target', dest='target', help='target model format')
    subparsers.required = True
    parser_onnx = subparsers.add_parser('onnx', help='export to ONNX')
    parser_openvino = subparsers.add_parser('openvino', help='export to OpenVINO')
    parser_openvino.add_argument('--input_shape', nargs=2, type=int, default=None,
                                 help='input shape as a height-width pair')
    parser_openvino.add_argument('--alt_ssd_export', action='store_true',
                                 help='use alternative ONNX representation of SSD net')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
