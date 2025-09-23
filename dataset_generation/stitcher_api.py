# import os
import requests

from inference.sahi_stitched import label_studio_to_coco

# local dev
# STITCHER_URL = 'http://localhost:8090'
# production url
STITCHER_URL = 'http://ecdysis01.local:8090'
FILE_MOUNT = '/pool1/srv/label-studio/mydata/stitchermedia'

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


def extract_bbox(a):
    return (a['x'], a['y'], a['width'], a['height'])


def filter_transform_record(row):
    if not row['annotations']:
        return

    # replace with FILE_MOUNT for production
    # use file_mount for local dev
    # cwd = os.getcwd()
    # file_mount = cwd.replace('ultralytics', 'label-studio/mydata/stitchermedia')
    file_name = row['panorama_path'].replace('/media', '')
    file_name = file_name.replace('/panorama', '_panorama')
    row['panorama_path'] = FILE_MOUNT + file_name
    coco_annotations = [
        label_studio_to_coco(
            extract_bbox(a), a['original_width'], a['original_height']) for a in row['annotations']]
    row.update({
        'coco_annotations': coco_annotations,
        'file_name': file_name
    })
    return row
