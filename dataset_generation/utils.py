import os
from pathlib import Path
import yaml


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


def check_missing_files(data):
    # check if files exist
    missingcsv = 'local_files/missing_images.csv'
    print('Checking for missing images ...')
    data['exists'] = data['full_image_path'].astype(str).map(os.path.exists)
    missing_images = data[data['exists'] == False]
    if len(missing_images):
        v = len(missing_images)
        if v >= 20:
            v = 20
        print('some images are missing. Up to the first 20 are...')
        print(missing_images.iloc[0:v])
        print('saving to file {0} ....'.format(missingcsv))
        missing_images.to_csv(missingcsv)
        return None
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
    if x >= 1.0:
        return True
    else:
        return False
