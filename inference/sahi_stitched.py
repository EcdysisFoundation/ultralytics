import os
import json
import requests

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# SAHI INFERENCE FOR OBJECT DETECTION

MODEL_PATH = 'runs/detect/train6/weights/best.pt'


detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=0.3,
    device='cuda:0'  # or 'cpu'
)


def format_result_label_studio(coco_result, image_width, image_height):

    result = [
                {
                    "id": str(i),
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "from_name": "label",
                    "to_name": "image",
                    "origin": "prediction",
                    "type": "rectanglelabels",
                    "value": {
                            "x": v['bbox'][0] / image_width * 100,
                            "y": v['bbox'][1] / image_height * 100,
                            "width": v['bbox'][2] / image_width * 100,
                            "height": v['bbox'][3] / image_height * 100,
                            "rotation": 0,
                            "rectanglelabels": ["Arthropod"],
                        }
                } for i, v in enumerate(coco_result)
            ]
    return json.dumps(result)


def label_studio_to_coco(bbox, image_width, image_height):
    """
    Undo format_result_label_studio() bounding box.
    """
    x, y, width, height = bbox
    x = x / 100 * image_width
    y = y / 100 * image_height
    width = width / 100 * image_width
    height = height / 100 * image_height
    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'image_width': image_width,
        'image_height': image_height
    }


def predict(img_path, save_img_file=False):
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

    return format_result_label_studio(
        coco_result, result.image_width, result.image_height)


def put_predictions(stitcher_url, guid, predictions):

    params = {'guid': str(guid)}
    api_post_url = stitcher_url + 'update-predictions/'

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_post_url, params=params, data=predictions, headers=headers)
        if response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        else:
            print('Response returned None')
    except Exception as e:
        print(e)
