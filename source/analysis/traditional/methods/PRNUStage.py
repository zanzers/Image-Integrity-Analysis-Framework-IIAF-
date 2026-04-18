import cv2
import numpy as np
from ...interfaces import IAnalysisStage, StageResult



class PRNUStage(IAnalysisStage):
    def __init__(self):
        self.numerator_sum = None
        self.denominator_sum = None
        self.image_count = 0

    @property
    def name(self) -> str:
        return "PRNU"

    def process(self, input_img: np.ndarray) -> StageResult:
        i_out = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        i_in = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21).astype(np.float32) / 255.0
        w = i_out - i_in
        num = w * i_in
        if self.numerator_sum is None:
            self.numerator_sum = np.zeros_like(num)
            self.denominator_sum = np.zeros_like(num)
        self.numerator_sum += num
        self.denominator_sum += i_in ** 2
        self.image_count += 1
        eps = 1e-8
        denom_safe = self.denominator_sum + eps
        f = self.numerator_sum / denom_safe
        f = np.nan_to_num(f, nan=0)
        mean = np.mean(f, axis=(0,1))
        f_zero_mean = f - mean
        sq = f_zero_mean ** 2
        norm = np.sqrt(np.sum(sq)) + 1e-8
        f_normalized = f_zero_mean / norm
        f_normalized = np.nan_to_num(f_normalized, nan=0)
        f_output = cv2.normalize(f_normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        features_flat = f_normalized.flatten()
        features_flat = np.nan_to_num(features_flat, nan=0, posinf=0, neginf=0)
        mean_out = np.mean(features_flat)
        stddev = np.std(features_flat)
        energy = np.linalg.norm(features_flat)
        pre_mean = np.mean(f_zero_mean)
        pre_std = np.std(f_zero_mean)

        features_prnu = {
            "PRNU_Mean": mean_out,
            "PRNU_Normal_Mean": pre_mean,
            "PRNU_Normal_Std": pre_std,
            "PRNU_StdDev": stddev,
            "PRNU_Energy": energy
        }
        return StageResult(f_output, features_prnu)
