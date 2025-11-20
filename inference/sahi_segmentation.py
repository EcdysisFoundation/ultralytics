import os

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# SAHI INFERENCE FOR SEGMENTATION


MODEL_PATH = 'bugmasker_weights.pt'


detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=0.3,
    device='cuda:0'  # or 'cpu'
)


def predict(img_path, save_img_file=False):
    print(f'running prediction on device {detection_model.device}')
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=2000,
        slice_width=2000,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    coco_result = result.to_coco_predictions(
        image_id=os.path.basename(img_path))

    # optionally save image file
    if save_img_file:
        result.export_visuals(
            export_dir="local_files/testing/",
            file_name='testfile',
            hide_labels=True,
            hide_conf=True)

    return coco_result
