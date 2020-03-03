import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (PseudoSampler, anchor_inside_flags, bbox2delta, keypoints2delta,
                        build_assigner, delta2bbox, delta2keypoints, force_fp32,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
from .atss_head import ATSSHead


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.reduce_op.SUM)
    return tensor


@HEADS.register_module
class ATSSLandmarksHead(ATSSHead):
    """
    Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_keypoints=5,
                 loss_keypoints=None,
                 **kwargs):

        self.num_keypoints = num_keypoints
        super().__init__(**kwargs)
        self.keypoints_loss = build_loss(loss_keypoints)

    def _init_layers(self):
        super()._init_layers()
        self.keypoints = nn.Conv2d(
            self.feat_channels, self.num_keypoints * 2, 3, padding=1)

    def init_weights(self):
        super().init_weights()
        normal_init(self.keypoints, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        keypoints = self.keypoints(reg_feat)
        return cls_score, bbox_pred, centerness, keypoints

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, keypoints,
                    labels, label_weights, bbox_targets, keypoints_targets, keypoints_valid, num_total_samples, cfg):

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        keypoints = keypoints.permute(0, 2, 3, 1).reshape(-1, self.num_keypoints, 2)
        keypoints_valid = keypoints_valid.unsqueeze(-1).unsqueeze(-1).expand_as(keypoints_targets).reshape(-1, self.num_keypoints, 2)
        keypoints_targets = keypoints_targets.reshape(-1, self.num_keypoints, 2)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        pos_inds = torch.nonzero(labels).squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            pos_keypoints_targets = keypoints_targets[pos_inds]
            pos_keypoints_pred = keypoints[pos_inds]
            pos_keypoints_valid = keypoints_valid[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = delta2bbox(pos_anchors, pos_bbox_pred,
                                              self.target_means,
                                              self.target_stds)
            pos_decode_bbox_targets = delta2bbox(pos_anchors, pos_bbox_targets,
                                                 self.target_means,
                                                 self.target_stds)

            pos_decode_keypoints_pred = delta2keypoints(pos_anchors, pos_keypoints_pred,
                                                        self.target_means,
                                                        self.target_stds)
            pos_decode_keypoints_targets = delta2keypoints(pos_anchors, pos_keypoints_targets,
                                                           self.target_means,
                                                            self.target_stds)

            loss_keypoints = self.keypoints_loss(pos_decode_keypoints_pred,
                                                 pos_decode_keypoints_targets,
                                                 pos_keypoints_valid)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = loss_cls * 0
            loss_centerness = loss_bbox * 0
            centerness_targets = torch.tensor(0).cuda()
            loss_keypoints = loss_cls * 0

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum(), loss_keypoints

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'keypoints'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             keypoints,
             gt_bboxes,
             gt_labels,
             gt_keypoints,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.atss_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_keypoints,
            img_metas,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, keypoints_targets_list, keypoints_valid_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor, loss_keypoints = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                keypoints,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                keypoints_targets_list,
                keypoints_valid_list,
                num_total_samples=num_total_samples,
                cfg=cfg)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,
            loss_keypoints=loss_keypoints)

    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        gts = delta2bbox(anchors, bbox_targets, self.target_means,
                         self.target_stds)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'keypoints_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   keypoints_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        from torch.onnx import is_in_onnx_export
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i] if is_in_onnx_export() else cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            keypoints_pred_list = [
                keypoints_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               centerness_pred_list,
                                               keypoints_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          keypoints_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_keypoints = []

        for cls_score, bbox_pred, centerness, anchors, keypoints_pred in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors, keypoints_preds):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            keypoints_pred = keypoints_pred.permute(1, 2, 0).reshape(-1, self.num_keypoints, 2)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                keypoints_pred = keypoints_pred[topk_inds]

            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            keypoints = delta2keypoints(anchors, keypoints_pred, self.target_means,
                                        self.target_stds)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_keypoints.append(keypoints)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_keypoints = torch.cat(mlvl_keypoints)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_keypoints /= mlvl_keypoints.new_tensor(scale_factor[:2])

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)

        det_bboxes, det_labels, det_keypoints = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_keypoints,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels, det_keypoints

    def atss_target(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_keypoints_list,
                    img_metas,
                    cfg,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """
        almost the same with anchor_target, with a little modification,
        here we need return the anchor
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_keypoints_targets, all_keypoints_valid, pos_inds_list, neg_inds_list) = multi_apply(
             self.atss_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             gt_keypoints_list,
             img_metas,
             cfg=cfg,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        keypoints_targets_list = images_to_levels(all_keypoints_targets, num_level_anchors)
        keypoints_valid_list = images_to_levels(all_keypoints_valid, num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list,
                keypoints_targets_list, keypoints_valid_list, num_total_pos,
                num_total_neg)

    def atss_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           gt_keypoints,
                           img_meta,
                           cfg,
                           label_channels=1,
                           unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        pos_gt_keypoints = gt_keypoints[sampling_result.pos_assigned_gt_inds, :]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        keypoints_targets = anchors.new_zeros((num_valid_anchors, self.num_keypoints, 2), dtype=torch.float)
        keypoints_valid = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                          sampling_result.pos_gt_bboxes,
                                          self.target_means, self.target_stds)
            pos_keypoints_targets, valid = keypoints2delta(sampling_result.pos_bboxes,
                                                           pos_gt_keypoints,
                                                           self.target_means, self.target_stds)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            keypoints_targets[pos_inds] = pos_keypoints_targets
            keypoints_valid[pos_inds] = valid
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            keypoints_targets = unmap(keypoints_targets, num_total_anchors, inside_flags)
            keypoints_valid = unmap(keypoints_valid, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, keypoints_targets, keypoints_valid,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
