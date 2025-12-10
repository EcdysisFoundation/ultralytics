import json
import os
import requests
from pathlib import Path

from dataset_generation.utils import (
    filter_transform_record,
    filter_transform_segmentation_record,
    FILE_MOUNT
)

# local dev
# STITCHER_URL = 'http://localhost:8090'
# production url
STITCHER_URL = 'http://ecdysis01.local:8090'

ERROR_MSG_KEY = 'ERROR'


def list_upload_files():
    api_list_url = STITCHER_URL + '/list-upload-files/'
    all_data = []
    offset = 0
    limit = 100

    while True:
        print('*'*100)
        params = {
            'offset': offset,
            'limit': limit,
            'approved': True
        }
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

    return all_data


def get_root_message():
    api_url = STITCHER_URL
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return {ERROR_MSG_KEY: response.status_code}
    except Exception as e:
        print(e)
        return {ERROR_MSG_KEY: e}


def pano_training_set():
    """
    Potentially broken, moved here after previous use.
    For object detection SAHI training set
    """

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


def pano_segmentation_training_set():
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
    not_done = True
    while not_done:
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
                # dev, pick one rec.
                if row['guid'] != 'c23201a2-1d9b-4232-bea4-7aaad383a253':
                    continue

                if row['annotations']:
                    original_width = row['annotations_segment'][0]['original_width']
                    original_height = row['annotations_segment'][0]['original_height']
                    image_id = len(coco_json_source["images"])
                    r = filter_transform_segmentation_record(row, image_id, original_width, original_height)
                    #dst = dataset_path / r['file_name']
                    #src = source_img_path / row['panorama_path'].replace('/media', '')

                    # temp for testing
                    file_name = str(row['guid']) + '_panorama.jpg'
                    # end temp for testing, get from row instead

                    #if src.is_file():
                    ##    if not dst.is_file():
                    #        dst.symlink_to(src)
                    #else:
                    #    print(f'WARNING: skipping missing img at {src}')
                    #    continue
                    coco_json_source['images'].append({
                        "height": original_height,
                        "width": original_width,
                        "id": image_id,
                        "file_name": file_name})

                    coco_json_source['annotations'] += r['coco_annotations']
                    not_done = False
                    if not not_done:
                        print('stopping early')
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    with open(out_json, 'w') as f:
        json.dump(coco_json_source, f)  # , indent=1
