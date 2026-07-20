import json
import os
import numpy as np
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from sahi.predict import get_sliced_prediction

from .sahi_segmentation import DETECTION_MODEL


FILE_LOCK = threading.Lock()
TEMP_RESULTS_PREFIX = 'temp_results_'


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy data types during JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def predict_and_stream(images_root, output_dir, save_img_file, img_entry, total_img_count):

    image_id = img_entry['id']
    file_name = img_entry['file_name']
    img_path = os.path.join(images_root, file_name)

    # Run Sliced Prediction
    result = get_sliced_prediction(
        img_path,
        DETECTION_MODEL,
        batch_size=10,
        slice_height=2000,
        slice_width=2000,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
        postprocess_match_threshold=0.3,
        perform_standard_pred=True,
    )

    # Convert to COCO format using the INTEGER ID from the GT
    coco_predictions = result.to_coco_predictions(image_id=image_id)

    # --- clean data ---
    cleaned_coco = []
    for pred in coco_predictions:
        # Skip predictions with no bounding box
        if not pred.get('bbox'):
            continue

        # Shift indexes by one (coco evaluation ignores zero index)
        pred['category_id'] = int(pred['category_id']) + 1
        cleaned_coco.append(pred)

    # skip if no annotations
    if cleaned_coco:
        if save_img_file:
            if (total_img_count % save_img_file) == 0:
                filename = Path(file_name).stem
                result.export_visuals(
                    export_dir=output_dir,
                    file_name=filename,
                    hide_labels=True,
                    hide_conf=True)

        thread_id = threading.get_ident()
        temp_filename = f"{output_dir}{TEMP_RESULTS_PREFIX}{thread_id}.jsonl"
        with open(temp_filename, "a") as f:
            f.write(json.dumps(cleaned_coco, cls=NumpyEncoder) + "\n")


def run_evaluation_inference(dataset_json, images_root, eval_output_file, output_dir, save_img_file):
    total_img_count = 0
    # Load the ground truth to get the correct Image IDs
    with open(dataset_json, 'r') as f:
        gt_data = json.load(f)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for img_entry in gt_data['images']:
            total_img_count += 1
            executor.submit(
                predict_and_stream,
                images_root, output_dir, save_img_file, img_entry, total_img_count)

    # Combine all temp files into one master file at the end
    print("Stitching files together...")
    with open(eval_output_file, "w") as master_file:
        for fname in os.listdir(output_dir):
            if fname.startswith(TEMP_RESULTS_PREFIX) and fname.endswith(".jsonl"):
                full_temp_path = os.path.join(output_dir, fname)
                with open(full_temp_path, "r") as temp_f:
                    master_file.write(temp_f.read())
                os.remove(full_temp_path)  # Clean up temp file


def convert_jsonl_to_coco_json(jsonl_input_path, json_output_path):
    combined_predictions = []

    print(f"Reading streaming results from {jsonl_input_path}...")

    with open(jsonl_input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            cleaned_line = line.strip()
            if not cleaned_line:
                continue  # Skip any accidental blank lines

            try:
                # Parse the line (which is a list of dicts for a single image)
                image_predictions = json.loads(cleaned_line)

                # Flatten the list into our master collection
                combined_predictions.extend(image_predictions)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Skipping malformed JSON on line {line_num}: {e}")

    print(f"Extracted {len(combined_predictions)} total bounding box/segmentation predictions.")
    print(f"Writing unified COCO JSON to {json_output_path}...")

    # Write out as a single standard JSON file
    with open(json_output_path, "w") as f:
        json.dump(combined_predictions, f)

    print("🎉 Conversion complete!")


if __name__ == "__main__":
    """
    eval_output_json intended as input for..
    sahi coco evaluate --dataset_json_path /home/ecdysis/ultralytics/eval_dataset_pano/dataset_test.json \
                   --result_json_path /home/ecdysis/ultralytics/eval_dataset_pano/evaluation_result.json \
                   --type segm \
                   --classwise
    """
    curr_dir = os.getcwd()
    dataset_dir = f'{curr_dir}/eval_dataset_pano/'
    dataset_json = f'{dataset_dir}dataset_test.json'
    images_root = f'{dataset_dir}images/test'
    eval_output_file = f'{dataset_dir}evaluation_result.jsonl'
    eval_output_json = f"{dataset_dir}evaluation_result.json"
    output_dir = "local_files/output/"
    save_img_file = 1  # save an image file every n images
    run_evaluation_inference(dataset_json, images_root, eval_output_file, output_dir, save_img_file)
    convert_jsonl_to_coco_json(eval_output_file, eval_output_json)
    print(f"Evaluation results saved to {eval_output_json}")
