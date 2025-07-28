import sys
import logging
import os
import pandas as pd
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

logger.setLevel(logging.INFO)


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
