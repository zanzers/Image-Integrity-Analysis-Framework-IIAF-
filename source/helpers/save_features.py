from typing import Dict, Optional
from .writer import FeatureJsonWriter


class SaveFeatures:
    @staticmethod
    def handle_save(combined_features: Dict[str, float], label: Optional[str] = None):
        if not combined_features:
            print("[WARN] No features to save.")
            return
        try:
            FeatureJsonWriter.append_features(combined_features, "features")
            print("[INFO] Combined features successfully saved to JSON.\n")
        except Exception as ex:
            print(f"[ERROR] Failed to save features: {ex}")