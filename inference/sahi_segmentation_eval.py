import json
import os
from pathlib import Path

from sahi.predict import get_sliced_prediction

from .sahi_segmentation import DETECTION_MODEL


def run_evaluation_inference(dataset_json, images_root, eval_output, save_img_file=False):
    """
    Intended as input for..
    sahi coco evaluate --dataset_json_path /home/ecdysis/ultralytics/local_files/evaluation_dataset_2/dataset_test.json \
                   --result_json_path /home/ecdysis/ultralytics/local_files/evaluation_dataset_2/evaluation_result.json \
                   --type segm \
                   --classwise
    """
    # Load the ground truth to get the correct Image IDs
    with open(dataset_json, 'r') as f:
        gt_data = json.load(f)

    all_coco_predictions = []

    # Loop through images defined in the Ground Truth
    for img_entry in gt_data['images']:
        image_id = img_entry['id']
        file_name = img_entry['file_name']
        img_path = os.path.join(images_root, file_name)

        # Run Sliced Prediction
        # batch_size=16 tells the GPU to process 16 slices at once
        result = get_sliced_prediction(
            img_path,
            DETECTION_MODEL,
            # batch_size=10, # in version 11.36
            slice_height=2000,
            slice_width=2000,
            overlap_height_ratio=0.4,
            overlap_width_ratio=0.4,
            postprocess_match_threshold=0.3,
            perform_standard_pred=True,
        )

        # Convert to COCO format using the INTEGER ID from the GT
        coco_predictions = result.to_coco_predictions(image_id=image_id)
        all_coco_predictions.extend(coco_predictions)

        if save_img_file:
            filename = Path(file_name).stem
            result.export_visuals(
            export_dir="local_files/output/",
            file_name=filename,
            hide_labels=True,
            hide_conf=True)

    #  data cleaning, mods
    #    Shift indexes by one because coco evaluation will ignore zero index.
    #    dataset_json should have already been shifted
    for pred in all_coco_predictions:
        pred['category_id'] = int(pred['category_id']) + 1
    # remove predictions with no bounding box
    all_coco_predictions = [v for v in all_coco_predictions if v['bbox']]

    # Save the final result.json
    with open(eval_output, 'w') as f:
        json.dump(all_coco_predictions, f)

    print(f"Evaluation results saved to {eval_output}")


if __name__ == "__main__":
    # insert paths, put result with dataset_json instead of somewhere else.
    dataset_dir = '/home/ecdysis/ultralytics/local_files/evaluation_dataset_1/'
    dataset_json = f'{dataset_dir}dataset_test.json'
    images_root = f'{dataset_dir}images/test'
    eval_output = f'{dataset_dir}evaluation_result.json'
    run_evaluation_inference(dataset_json, images_root, eval_output)
