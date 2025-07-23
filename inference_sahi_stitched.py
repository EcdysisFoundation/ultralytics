import os
import json
import requests

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# get the stitcher repo directory, assumes co-located
stitcher_dir = os.getcwd().replace('ultralytics', 'stitcher')

api_list_url = 'http://localhost:8090/list-upload-files/'
api_post_url = 'http://localhost:8090/update-predictions/'
model_path = 'runs/detect/train3/weights/best.pt'


detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model_path,
    confidence_threshold=0.3,
    device='cuda:0' # or 'cpu'
)


def format_result_label_studio(coco_result, image_width, image_height):

    result = [
                {
                    "id": i,
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "from_name": "label",
                    "to_name": "image",
                    "origin": "prediction",
                    "type": "rectanglelabels",
                    "value": {
                            "x": v['bbox'][0],
                            "y": v['bbox'][1],
                            "width": v['bbox'][2],
                            "height": v['bbox'][3],
                            "rotation": 0,
                            "rectanglelabels": ["Arthropod"],
                        }
                } for i, v in enumerate(coco_result)
            ]
    return json.dumps(result)


def predict(img_path, save_img_file=False):
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    coco_result = result.to_coco_predictions(image_id=os.path.basename(img_path))

    # optionally save image file
    if save_img_file:
        result.export_visuals(
            export_dir="local_files/testing/",
            file_name='testfile',
            hide_labels=True,
            hide_conf=True)


    return format_result_label_studio(coco_result, result.image_width, result.image_height)


def put_predictions(guid, predictions):

    params = {'guid': str(guid)}

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


def main():
    all_data = []
    offset =  0
    limit = 100


    while True:
        print('*'*100)
        params = {'offset': offset, 'limit': limit}
        print(params)

        try:
            response = requests.get(api_list_url, params=params)
        except Exception as e:
            print(e)
            break

        if response.status_code == 200:
            data = response.json()
            if not data:
                break

            all_data.extend(data)
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    print(f"Retrieved {len(all_data)} items.")

    # counter = 0 # stop early while testing
    for d in all_data:
        # if counter == 1:
        #    break
        if d['panorama_path']:
            p = stitcher_dir + d['panorama_path']
            if os.path.exists(p):
                predictions = predict(p)
                put_predictions(d['guid'], predictions)
                counter += 1
            else:
                print('panorama file does not exist: {0}'.format(p))


if __name__ == '__main__':
    main()
