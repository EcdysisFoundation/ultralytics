import os

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# SAHI INFERENCE FOR SEGMENTATION


MODEL_PATH = 'runs/segment/train2/weights/best.pt'


DETECTION_MODEL = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=0.3,
    device='cuda:0'  # or 'cpu'
)


def predict(img_path, save_img_file=False):
    print(f'running prediction on device {DETECTION_MODEL.device}')
    result = get_sliced_prediction(
        img_path,
        DETECTION_MODEL,
        slice_height=2000,
        slice_width=2000,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
        postprocess_match_threshold=0.3,
        perform_standard_pred=True
    )
    original_width = result.image_width
    original_height = result.image_height
    coco_result = result.to_coco_predictions(
        image_id=os.path.basename(img_path))
    # optionally save image file
    if save_img_file:
        combined_path_filename = img_path.replace('/panorama', '_panorama')
        filename = os.path.splitext(os.path.basename(combined_path_filename))[0]
        result.export_visuals(
            export_dir="local_files/output/inference",
            file_name=filename,
            hide_labels=True,
            hide_conf=True)

    return coco_result, original_width, original_height
