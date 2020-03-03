from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class SingleStageDetectorLm(SingleStageDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False, postprocess=True):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        det_bboxes, det_labels, det_keypoints = \
            self.bbox_head.get_bboxes(*outs, img_meta, self.test_cfg, False)[0]

        if postprocess:
            return self.postprocess(det_bboxes, det_labels, det_keypoints, None, img_meta,
                                    rescale=rescale)

        return det_bboxes, det_labels, det_keypoints
