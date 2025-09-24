import os
import json
import requests
import sys
import argparse
import logging
from pathlib import Path

from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from ultralytics.data.converter import convert_coco

from PIL import Image

from .split import split_from_df, split_by_labels_train_val, DATASETS_FOLDER
from .stitcher_api import (
    filter_transform_record, get_root_message,
    ERROR_MSG_KEY, STITCHER_URL, FILE_MOUNT)
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
    curr_dir = os.getcwd()
    print(f'curr_dir: {curr_dir}')
    curr_dir = os.getcwd()
    dataset_dir = curr_dir + '/dataset_pano'
    out_json = dataset_dir + '/dataset.json'
    print(f'out_json: {out_json}')
    dataset_path = Path(dataset_dir)
    source_img_dir = FILE_MOUNT
    print(f'source_img_dir: {source_img_dir}')
    source_img_path = Path(source_img_dir)

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
        print('-list-upload-files-' * 6)
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
            for row in data:
                if row['annotations']:
                    r = filter_transform_record(row)
                    dst = dataset_path / r['file_name']
                    src = source_img_path / row['panorama_path'].replace('/media', '')
                    if src.is_file():
                        if not dst.is_file():
                            dst.symlink_to(src)
                    else:
                        print(f'WARNING: skipping missing img at {src}')
                        continue
                    coco_json_source['images'].append({
                        "height": r['coco_annotations'][0]['image_height'],
                        "width": r['coco_annotations'][0]['image_width'],
                        "id": int(r['id']),
                        "file_name": r['file_name']})
                    annotations = [{
                        "category_id": 1,
                        "image_id": int(r['id']),
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
        json.dump(coco_json_source, f)  # , indent=1


def slice_pano_training_set():
    curr_dir = os.getcwd()
    dataset_dir = curr_dir + '/dataset_pano'
    dataset_json_path = dataset_dir + '/dataset.json'
    dataset_sliced = dataset_dir + '/sliced/'
    print(f'dataset_file_path: {dataset_json_path}')

    coco_dict = load_json(dataset_json_path)
    print('coco_dict read, first image is')
    print(coco_dict["images"][0]["file_name"])

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    slice_coco(
        coco_annotation_file_path=dataset_json_path,
        image_dir=dataset_dir,
        output_coco_annotation_file_name="sliced_coco.json",
        ignore_negative_samples=False,
        output_dir=dataset_sliced,
        slice_height=2000,
        slice_width=2000,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.1,
        verbose=True
    )
    print('slice_pano_training_set done')


# run with `python -m dataset_generation`
if __name__ == '__main__':
    """
    Assumes running from ultralytics home dir with 'python -m dataset_generation'
    """
    pano_training_set()
    slice_pano_training_set()
    convert_coco("dataset_pano/sliced/", cls91to80=False, save_dir="dataset_pano/coco_converted")
    split_by_labels_train_val('dataset_pano/coco_converted/labels/sliced_coco.json_coco', 'dataset_pano/sliced')
