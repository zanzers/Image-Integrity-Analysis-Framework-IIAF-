import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from analysis.traditional.methods.ELASTAGE import ELAStage
from analysis.traditional.methods.IWTStage import IWTStage
from analysis.traditional.methods.PRNUStage import PRNUStage
from analysis.neural_networks.CNNStage import CnnStage
from analysis.interfaces import StageResult
from helpers.writer import FeatureExcelWriter


LABEL_MAP = {
    "Original": 0,
    "Faceswap": 1,
    "AI Content Insertion": 2,
    "Deepfake": 3
}

def extract_features(image_path: Path) -> Optional[Dict[str, float]]:
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None


    ela_stage = ELAStage()
    iwt_stage = IWTStage()
    prnu_stage = PRNUStage()
    # cnn_stage = CnnStage()

    results = []
    results.append(stage_process(ela_stage, image))
    results.append(stage_process(iwt_stage, image))
    results.append(stage_process(prnu_stage, image))
    # results.append(stage_process(cnn_stage, image))

    combined_features = {}
    for result in results:
        if result['success'] and result['features']:
            combined_features.update(result['features'])

    return combined_features


def stage_process(stage, image: np.ndarray) -> Dict:
    """Process a single stage and return result dict."""
    stage_name = stage.name
    try:
        result: StageResult = stage.process(image)
        if result.success:
            return {
                'stage_name': stage_name,
                'success': True,
                'features': result.features,
                'output_image': result.output_image,
                'message': result.output
            }
        else:
            print(f"{stage_name} analysis failed: {result.output}")
            return {
                'stage_name': stage_name,
                'success': False,
                'features': {},
                'output_image': None,
                'message': result.output or "Analysis failed"
            }
    except Exception as e:
        print(f"Error in {stage_name}: {e}")
        return {
            'stage_name': stage_name,
            'success': False,
            'features': {},
            'output_image': None,
            'message': str(e)
        }

def main():
    
    uploads_datasets = Path("uploads_datasets")
    if not uploads_datasets.exists():
        print(f"Directory {uploads_datasets} does not exist.")
        return

    image_extensions = {'.jpg', '.jpeg', '.png'}

    for folder in uploads_datasets.iterdir():
        if not folder.is_dir():
            continue

        label_name = folder.name
        if label_name.isdigit():
            label = int(label_name)
        elif label_name in LABEL_MAP:
            label = LABEL_MAP[label_name]
        else:
            print(f"Unknown label folder: {label_name}. Skipping.")
            continue
        print(f"Processing folder: {label_name} (label: {label})")

        for img_file in folder.rglob('*'):
            if img_file.suffix.lower() in image_extensions:
                print(f"Processing image: {img_file}")
                features = extract_features(img_file)
                if features:
                    FeatureExcelWriter.append_features(features, str(label))
                else:
                    print(f"Failed to extract features from {img_file}")

    print(f"Processing complete. Features saved to {uploads_datasets}")
  

if __name__ == '__main__':
    main()