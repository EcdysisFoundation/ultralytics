import sys
import logging
import numpy as np
import os
import pandas as pd
import yaml

from pathlib import Path
from uuid import uuid4

from inference.sahi_stitched import label_studio_to_coco


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

logger.setLevel(logging.INFO)

FILE_MOUNT = '/pool1/srv/label-studio/mydata/stitchermedia'


def make_yaml_dict(dataset_folder, class_index):
    return {
            'path': '../' + dataset_folder,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': class_index
        }


def save_yaml_file(dataset_folder, class_names):
    yaml_name = 'data.yaml'
    y = make_yaml_dict(dataset_folder, class_names)
    with (Path(dataset_folder)/yaml_name).open('w') as f:
        yaml.dump(y, f)


def check_missing_files(data, test_flag):
    # check if files exist
    missingcsv = 'local_files/missing_images.csv'
    logger.info('Checking for missing images ...')
    data['exists'] = data['full_image_path'].astype(str).map(os.path.exists)
    missing_images = data[data['exists'] == False]
    if len(missing_images):
        v = len(missing_images)
        if v >= 20:
            v = 20
        logger.info('some images are missing. Up to the first 20 are...')
        logger.info(missing_images.iloc[0:v])
        logger.info('saving to file {0} ....'.format(missingcsv))
        missing_images.to_csv(missingcsv)
        if not test_flag:
            return None
        else:
            return 'In Testing mode, we found missing images...'
    return 'All images found'


def convert_annotation_to_yolo(labels):
    """
    modified from:
    https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/utils.py

    Convert LS annotation to Yolo format.

    Args:
        label (dict): Dictionary containing annotation information including:
            - width (float): Width of the object.
            - height (float): Height of the object.
            - x (float): X-coordinate of the top-left corner of the object.
            - y (float): Y-coordinate of the top-left corner of the object.

    Returns:
        tuple or None: If the conversion is successful, returns a tuple (x, y, w, h) representing
        the coordinates and dimensions of the object in Yolo format, where (x, y) are the center
        coordinates of the object, and (w, h) are the width and height of the object respectively.
    """
    result = []
    for label in labels:
        if ("x" in label and "y" in label and 'width' in label and 'height' in label):
            w = label['width']
            h = label['height']

            x = (label['x'] + w / 2) / 100
            y = (label['y'] + h / 2) / 100
            w = w / 100
            h = h / 100

            result.append([x, y, w, h])

    return result


def check_minimum_length(image_list, train_size):
    x = len(image_list) * train_size / 2
    if x >= 2.0:
        return True
    else:
        return False


def get_count_per_class_split(splits, class_name):
    """
    Get the number of images per class in each split
    splits has the following format (as in the splits.yaml file)
    {
     '99': {
            'test': [  '/path/to/test_image1_for_class_99.jpg',... ],
            'train': [  '/path/to/train_image1_for_class_99.jpg',...  ],
            'val': [  '/path/to/val_image1_for_class_99.jpg', ...]
        },
        ...
    }
    Args:
        splits: Dictionary of lists of image paths per split, the key is the class name, the value is a dict of split, list of image path of that split and class
    Returns:
        Dataframe with the number of images per class in each split, columns are split names (train,test,val), rows are class ids
    """
    counts = []

    for class_id, split in splits.items():
        # id, train, test, val
        counts.append({class_name:class_id, **{split_name: len(image_paths) for split_name, image_paths in split.items()}})
    return pd.DataFrame(counts)


def generate_split_class_report(splits, class_name):
    """
    Return the dataset sample count report
    """

    counts_df = get_count_per_class_split(splits, class_name)

    counts_df["total_samples"] = counts_df["train"] + counts_df["val"] + counts_df["test"]

    return counts_df.sort_values(by=class_name)


def convert_coco_to_yolo(c):
    """
    Convert coco annotation to yolo format given
    'x': x,
    'y': y,
    'width': width,
    'height': height,
    'image_width': image_width,
    'image_height': image_height
    return x_center y_center width height # as ratios
    """
    x_center = (c['width'] / 2 + c['x']) / c['image_width']
    y_center = (c['height'] / 2 + c['y']) / c['image_height']
    width = c['width'] / c['image_width']
    height = c['height'] / c['image_height']

    return (x_center, y_center, width, height)


def get_polygon_area(x, y):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/utils.py
    https://en.wikipedia.org/wiki/Shoelace_formula

    """

    assert len(x) == len(y)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/utils.py
    """

    assert len(x) == len(y)
    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def convert_ls_polygonlabels_to_coco(
        annotation_id, image_id,
        points, width, height):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/converter.py#L836
    """
    points_abs = [
        (x / 100 * width, y / 100 * height) for x, y in points
    ]
    x, y = zip(*points_abs)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,  # single category
        "segmentation":
            [
                [coord for point in points_abs for coord in point]
            ],
        "bbox": get_polygon_bounding_box(x, y),
        "ignore": 0,
        "iscrowd": 0,
        "area": get_polygon_area(x, y),
    }


def convert_coco_segmentation_to_ls(
    category_id, segmentation, categories, from_name, image_height, image_width, to_name
):
    """
    Modified from https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/imports/coco.py
     function name = create_segmentation
    Convert COCO segmentation annotation to Label Studio polygon format.

    COCO segmentation format: flat array of [x1,y1,x2,y2,...] coordinates
    Label Studio format: array of [x,y] points as percentages

    Args:
        category_id (int): COCO category ID for this segmentation
        segmentation (list): Flat list of polygon coordinates [x1,y1,x2,y2,...]
        categories (dict): Mapping of category_id to category name
        from_name (str): Control tag name from Label Studio labeling config
        image_height (int): Height of the source image in pixels
        image_width (int): Width of the source image in pixels
        to_name (str): Object name from Label Studio labeling config

    Returns:
        dict: Label Studio polygon annotation item
    """
    label = categories[int(category_id)]
    # Convert flat array [x1,y1,x2,y2,...] to array of points [[x1,y1],[x2,y2],...]
    points = [list(x) for x in zip(*[iter(segmentation)] * 2)]

    # Convert absolute coordinates to percentages
    for i in range(len(points)):
        points[i][0] = points[i][0] / image_width * 100.0
        points[i][1] = points[i][1] / image_height * 100.0

    item = {
        "id": uuid4().hex[0:10],
        "type": "polygonlabels",
        "value": {"points": points, "polygonlabels": [label]},
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
    }
    return item


def extract_bbox(a):
    return (a['x'], a['y'], a['width'], a['height'])


def filter_transform_record(row):
    """
    For object detections, make coco annotations and provide image path.
    """
    if not row['annotations']:
        return

    # replace with FILE_MOUNT for production
    # use file_mount for local dev
    # cwd = os.getcwd()
    # file_mount = cwd.replace('ultralytics', 'label-studio/mydata/stitchermedia')
    file_name = row['panorama_path'].replace('/media/', '')
    file_name = file_name.replace('/panorama', '_panorama')
    panorama_path = row['panorama_path'].replace('/media', '')
    row['panorama_path'] = FILE_MOUNT + panorama_path
    coco_annotations = [
        label_studio_to_coco(
            extract_bbox(a), a['original_width'], a['original_height']) for a in row['annotations']]
    row.update({
        'coco_annotations': coco_annotations,
        'file_name': file_name
    })
    return row


def filter_transform_segmentation_record(row, image_id, width, height):
    """
    For segmentation, make coco annotations and provide image path.
    """
    if not row['annotations_segment']:
        return

    coco_annotations = [convert_ls_polygonlabels_to_coco(
        i, image_id,
        v['points'], width, height) for i, v in enumerate(row['annotations_segment'])]
    row.update({
        'coco_annotations': coco_annotations,
    })
    return row
