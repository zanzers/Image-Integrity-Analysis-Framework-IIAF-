import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from ...interfaces import IAnalysisStage, StageResult


class ELAStage(IAnalysisStage):
    @property
    def name(self) -> str:
        return "ELA"

    def process(self, input_img: np.ndarray) -> StageResult:

        input_float = input_img.astype(np.float32) / 255.0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

        _, jpeg_bytes = cv2.imencode('.jpg', input_img, encode_param)
        jpeg_img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        ela = cv2.absdiff(input_float, jpeg_img)
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
        data = ela_gray.flatten().astype(np.float64)
        mean = np.mean(data)
        variance = np.var(data)
        stddev = np.sqrt(variance)
        bins = 256
        hist, _ = np.histogram(data, bins=bins, range=(0, 1))
        hist = hist / len(data)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        skewness = skew(data)
        kurt = kurtosis(data)
        ela_output = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        features = {
            "Ela_Mean": mean,
            "Ela_StdDev": stddev,
            "Ela_Entropy": entropy,
            "Ela_Skewness": skewness,
            "Ela_Kurtosis": kurt
        }
        return StageResult(ela_output, features)
