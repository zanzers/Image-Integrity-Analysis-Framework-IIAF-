import cv2
import numpy as np
from ...interfaces import IAnalysisStage, StageResult



class IWTStage(IAnalysisStage):
    @property
    def name(self) -> str:
        return "IWT"

    def process(self, input_img: np.ndarray) -> StageResult:
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rows, cols = gray.shape
        rows_even = rows - (rows % 2)
        cols_even = cols - (cols % 2)
        resized = gray[:rows_even, :cols_even]
        ll = np.zeros((rows_even // 2, cols_even // 2), dtype=np.float32)
        lh = np.zeros_like(ll)
        hl = np.zeros_like(ll)
        hh = np.zeros_like(ll)
        for y in range(0, rows_even, 2):
            for x in range(0, cols_even, 2):
                a = resized[y, x]
                b = resized[y, x + 1]
                c = resized[y + 1, x]
                d = resized[y + 1, x + 1]
                ll[y // 2, x // 2] = (a + b + c + d) / 4
                lh[y // 2, x // 2] = (a + b - c - d) / 4
                hl[y // 2, x // 2] = (a - b + c - d) / 4
                hh[y // 2, x // 2] = (a - b - c + d) / 4
        ll_stats = self._compute_stats(ll)
        lh_stats = self._compute_stats(lh)
        hl_stats = self._compute_stats(hl)
        hh_stats = self._compute_stats(hh)
        top = np.hstack([ll, lh])
        bottom = np.hstack([hl, hh])
        output = np.vstack([top, bottom])
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        features = {
            "IWT_LL_Mean": ll_stats[0],
            "IWT_LL_Var": ll_stats[1],
            "IWT_LL_Entropy": ll_stats[2],
            "IWT_LH_Mean": lh_stats[0],
            "IWT_LH_Var": lh_stats[1],
            "IWT_LH_Entropy": lh_stats[2],
            "IWT_HL_Mean": hl_stats[0],
            "IWT_HL_Var": hl_stats[1],
            "IWT_HL_Entropy": hl_stats[2],
            "IWT_HH_Mean": hh_stats[0],
            "IWT_HH_Var": hh_stats[1],
            "IWT_HH_Entropy": hh_stats[2]
        }
        return StageResult(output, features)

    def _compute_stats(self, m: np.ndarray) -> list:
        mean_val = np.mean(m)
        var_val = np.var(m)
        hist, _ = np.histogram(m.flatten(), bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return [mean_val, var_val, entropy]
