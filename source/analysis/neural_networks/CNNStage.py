import os
import json
import tempfile
from PIL import Image
import cv2
from ..interfaces import *
from .model.cnn_model import ResNetClassifier


class CnnStage(IAnalysisStage):
    @property
    def name(self) -> str:
        return "CNN_Model"

    def process(self, input_img: np.ndarray) -> StageResult:
        try:
         
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                pil_img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
                pil_img.save(temp_path)
            
            classifier = ResNetClassifier(mode="fine_tuned")
            result = classifier.predict(temp_path)
            
            features_cnn = {
                "CNN_Label": result["label"],
                "CNN_Confidence": result["confidence"],
                "CNN_Prob_0": result["probs"][0],
                "CNN_Prob_1": result["probs"][1],
                "CNN_Prob_2": result["probs"][2],
                "CNN_Prob_3": result["probs"][3] if len(result["probs"]) > 3 else 0
            }
            
            os.unlink(temp_path) 
            
            return StageResult(features=features_cnn, output="[CNN Stage Completed]", success=True)
        except Exception as ex:
            print(f"[CNN model] Error: {ex}")
            return StageResult(output=str(ex), success=False)
