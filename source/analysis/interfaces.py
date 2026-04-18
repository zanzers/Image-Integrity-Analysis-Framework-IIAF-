from typing import Dict, Optional
import numpy as np
from enum import Enum


class StageResult:
    def __init__(self, output_image: Optional[np.ndarray] = None, features: Optional[Dict[str, float]] = None, output: Optional[str] = None, success: bool = True):
        self.output_image = output_image
        self.features = features or {}
        self.output = output
        self.success = success

class IAnalysisStage:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def process(self, input_img: np.ndarray) -> StageResult:
        raise NotImplementedError