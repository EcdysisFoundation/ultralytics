import os
import json
import requests
import sys
import argparse
import logging
from pathlib import Path
from .split import split_from_df, DATASETS_FOLDER
from .stitcher_api import (
    filter_transform_record, get_root_message,
    ERROR_MSG_KEY, STITCHER_URL)
from .data import ObjectDetectData
from .utils import convert_annotation_to_yolo, check_missing_files, generate_split_class_report

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument(
        '--class-col', type=str, default='specimen__classification__gbif_order',
        help='The column to catagorize the images')
    parser.add_argument('-t', '--test-flag', action='store_true')
    parser.add_argument('-cpy', '--copy-files', action='store_true')

    return parser.parse_args()


def single_specimen_trainingset():
    args = get_args()

    db = ObjectDetectData()
    full_data = db.get_full_df()

    category_counts = full_data[args.class_col].value_counts()

    logger.info('category counts')
    logger.info(category_counts)

    check_ok = check_missing_files(full_data, args.test_flag)

    if check_ok:
        print(check_ok)
    else:
        print('exiting...........')
        return

    full_data = db.get_full_df()
    full_data['yolo_annotations'] = full_data['object_det_label'].apply(convert_annotation_to_yolo)
    full_data.to_csv('local_files/full_data.csv')

    splits = split_from_df(full_data, args)
    report_count_df = generate_split_class_report(splits, args.class_col)
    report_count_df.to_csv(Path(DATASETS_FOLDER) / 'dataset_report.csv', index=False)

    print('end of main')


def pano_training_set():

    api_ping = get_root_message()
    print(api_ping)
    if ERROR_MSG_KEY in api_ping.keys():
        return

    api_list_url = STITCHER_URL + '/list-upload-files/'
    offset = 0
    limit = 10
    dataset_dir = os.getcwd()
    print('dataset_dir')
    print(dataset_dir)
    out_json = dataset_dir + '/dataset_pano/dataset.json'
    print('out_json')
    print(out_json)

    coco_json_source = {
        "images": [],
        "categories": [{
            "supercategory": "Arthropod",
            "id": 1,
            "name": "arthropod"}],
        "annotations": [],
    }

    while True:
        params = {
            'offset': offset,
            'limit': limit,
            'approved': True
        }
        print('-list-upload-files-' * 10)
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

            print(f'data returned from api for next {limit} records')
            for i, row in enumerate(data):
                if row['annotations']:
                    # temp limit to one record
                    if row['guid'] == '134854c0-f889-4933-9139-3d77f201be85':
                        r = filter_transform_record(row)
                        coco_json_source['images'].append({
                            "height": r['coco_annotations'][0]['image_height'],
                            "width": r['coco_annotations'][0]['image_width'],
                            "id": i,
                            "file_name": r['file_name']})
                        annotations = [{
                            "category_id": 1,
                            "image_id": i,
                            "bbox": (v['x'], v['y'], v['width'], v['height']),
                            "iscrowd": 0,
                            "segmentation": [],
                            "area": None
                        } for v in row['coco_annotations']]
                        coco_json_source['annotations'] += annotations

            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    with open(out_json, 'w') as f:
        json.dump(coco_json_source, f, indent=1)


# run with `python -m dataset_generation`
if __name__ == '__main__':
    """
    Assumes running from ultralytics home dir with 'python -m dataset_generation'
    """
    pano_training_set()
