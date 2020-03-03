from ..registry import DETECTORS
from .single_stage_lm import SingleStageDetectorLm


@DETECTORS.register_module
class ATSSLm(SingleStageDetectorLm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
