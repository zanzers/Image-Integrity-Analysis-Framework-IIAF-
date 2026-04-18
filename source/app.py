from pathlib import Path
import cv2
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from source.analysis.traditional.methods.ELASTAGE import ELAStage
from source.analysis.traditional.methods.IWTStage import IWTStage
from source.analysis.traditional.methods.PRNUStage import PRNUStage
from source.analysis.neural_networks.CNNStage import CnnStage
from source.helpers.save_features import SaveFeatures



def process_image(image_path: Path) -> bool:
    print(f"Processing image: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False

    print(f"Image loaded successfully. Shape: {image.shape}")

    try:
        # Initialize 
        ela_stage = ELAStage()
        iwt_stage = IWTStage()
        prnu_stage = PRNUStage()
        cnn_stage = CnnStage()


        results = []
        results.append(stageProcess(ela_stage, image))
        results.append(stageProcess(iwt_stage, image))
        results.append(stageProcess(prnu_stage, image))
        results.append(stageProcess(cnn_stage, image))

        combined_features = {}
        for result in results:
            if result['success'] and result['features']:
                combined_features.update(result['features'])

        SaveFeatures.handle_save(combined_features)

        successful_stages = sum(1 for r in results if r['success'])
        total_features = sum(len(r['features']) for r in results)

        print(f"\nProcessing complete: {successful_stages}/{len(results)} stages successful")
        for r in results:
            status = "SUCCESS" if r['success'] else "FAILED"
            print(f"  - {r['stage_name']}: {status} ({len(r['features'])} features)")
        print(f"Total features extracted: {total_features}")


        return successful_stages > 0

    except ImportError as e:
        print(f"Import error: {e}")
        print("Analysis stages could not be imported. Check dependencies.")
        return False
    except Exception as e:
        print(f"Processing error: {e}")
        return False

    


def stageProcess(stage, image: np.ndarray) -> dict:
    stage_name = stage.name
    print(f"Running {stage_name} analysis...")

    try:
        result = stage.process(image)

        if result.success:
            print(f"{stage_name} completed. Features extracted: {len(result.features)}")
            print(f"Sample {stage_name} features: {dict(list(result.features.items())[:3])}")

            return {
                'stage_name': stage_name,
                'success': True,
                'features': result.features,
                'output_image': result.output_image,
                'message': result.output
            }
        else:
            print(f"{stage_name} analysis failed")
            return {
                'stage_name': stage_name,
                'success': False,
                'features': {},
                'output_image': None,
                'message': result.output or "Analysis failed"
            }

    except Exception as e:
        print(f"{stage_name} analysis error: {e}")
        return {
            'stage_name': stage_name,
            'success': False,
            'features': {},
            'output_image': None,
            'message': str(e)
        }

