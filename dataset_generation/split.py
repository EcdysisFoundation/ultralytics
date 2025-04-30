import os
import logging
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
from shutil import copy, SameFileError
from tqdm import tqdm

from .utils import save_yaml_file, check_minimum_length


SEED = 42
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DATASETS_FOLDER = 'datasets'


def create_clear_dirs():
    parent_images = Path(DATASETS_FOLDER) / 'images'
    parent_labels = Path(DATASETS_FOLDER) / 'labels'

    # Clear previous runs, make fresh directories
    if os.path.exists(parent_images):
        shutil.rmtree(parent_images)
    if os.path.exists(parent_labels):
        shutil.rmtree(parent_labels)

    subfolders = ('train', 'val', 'test')
    for name in subfolders:
        i =  parent_images / name
        i.mkdir(parents=True)
        l = parent_labels / name
        l.mkdir(parents=True)

    return {
        'parent_images': parent_images,
        'parent_labels': parent_labels
    }


def save_class_images(splits: dict, c: str, df, class_to_index, dirs, args):
    """
    Save images of a class divided in splits
    This assumes single specimen images, one species per image
    Args:
        splits: Dictionary of lists of image paths per split
        c: Name of the class
        df: complete dataframe of records
        class_to_index: lookup to get index from class name
    """

    def copy_img(src: Path, dst: Path):
        logger.debug(f'Copying {src} to {dst}')
        try:
            copy(src, dst, follow_symlinks=True)
        except SameFileError:
            logger.warning(f'File {dst} already present, skipping')

    for split_name, split_img in splits[c].items():
        if len(split_img) == 0:
            continue

        parent_i =  dirs['parent_images'] / split_name
        parent_l = dirs['parent_labels'] / split_name

        logger.info(f'Writing images to {parent_i}')
        for img in tqdm(split_img,
                        desc=f'Copying {len(split_img)} {split_name} images of {c.replace("_", " ")} class'):
            src = Path(img)
            dst = parent_i / src.name
            label_filename = os.path.splitext(src.name)[0] + '.txt'

            # there should be onlyone here, take the first
            v = df[df['full_image_path'] == img].iloc[0]

            c_indx = class_to_index[v['specimen__classification__gbif_order']]

            if not args.test_flag:
                if args.copy_files:
                    copy_img(src, dst)
                else:
                    # Ultralytics does not currently support symlinks
                    # sourced on a different machine, if image.read() != b'\xff\xd9'
                    dst.symlink_to(src)

            # save the annotations label file
            with open(parent_l / label_filename, 'w') as f:
                for a in v['yolo_annotations']:
                    annotation = [c_indx] + a
                    for idx, l in enumerate(annotation):
                        if idx == len(annotation) - 1:
                            f.write(f"{l}\n")
                        else:
                            f.write(f"{l} ")


def split_from_df(
        df: pd.DataFrame,
        args,
        train_size=0.8):
    """
    Split images of a dataset in train/val/test. The splitting preserves the distribution of samples per class in each
    group (stratification).
    Args:
        df: Input DataFrame, the output of `db.get_reviewed_images`
        train_size: Proportion of images reserved for train. Val/Test sizes are computed as (1 - train_size)/2
        output: Path to output directory
        save_yaml: Create yaml splits file
        seed: Random state
        **kwargs: For yaml file name pass `yaml_name` as keyword argument
    """
    logger.info('running splits from df')
    if not 0.0 < train_size <= 1.0:
        raise ValueError('Train size must be between 0 and 1')

    df=df.copy()

    df.replace('', np.nan, inplace=True)  # Handle empty strings
    classes = df[args.class_col].drop_duplicates()
    class_index = {i: n for i, n in enumerate(classes)}
    class_to_index = {n: i for i, n in class_index.items()}

    images = dict(df.groupby(args.class_col)['full_image_path'].apply(list))
    dirs = create_clear_dirs()
    splits = {}
    for c, image_list in images.items():
        c = str(c)
        if not check_minimum_length(image_list, train_size):
            print('Not enough images for class: {0}, skipping this one'.format(c))
            continue
        train, test_val = train_test_split(image_list, train_size=train_size, random_state=SEED)
        val, test = train_test_split(test_val, train_size=train_size, random_state=SEED)

        splits[c] = {'train': train, 'val': val, 'test': test}

        save_class_images(splits, c, df, class_to_index, dirs, args)

    save_yaml_file(DATASETS_FOLDER, class_index)
    return splits
