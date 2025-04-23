import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from .utils import save_yaml_file


SEED = 42
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DATASETS_FOLDER = 'datasets'


def save_class_images(splits: dict, c: str, df, class_to_index):
    """
    Save images of a class divided in splits
    This assumes single specimen images, one species per image
    Args:
        splits: Dictionary of lists of image paths per split
        c: Name of the class
        df: complete dataframe of records
        class_to_index: lookup to get index from class name
    """

    for split_name, split_img in splits[c].items():
        if len(split_img) == 0:
            continue
        parent_i = Path(DATASETS_FOLDER) / 'images' / split_name
        parent_l = Path(DATASETS_FOLDER) / 'labels' / split_name

        print(f'Writing images to {parent_i}')
        for img in tqdm(split_img,
                        desc=f'Copying {len(split_img)} {split_name} images of {c.replace("_", " ")} class'):
            src = Path(img)
            dst = parent_i / src.name
            label_filename = os.path.splitext(src.name)[0] + '.txt'

            v = df[df['image'] == img]
            # there should be onlyone here, take the first
            c_indx = class_to_index[v['specimen__classification__gbif_order'].iloc[0]]

            if not dst.is_file() and src.is_file():
               dst.symlink_to(src)

            # save the annotations label file
            with open(parent_l / label_filename, 'w') as f:
                for a in v['yolo_annotations'].iloc[0]:
                    annotation = [c_indx] + a
                    for idx, l in enumerate(annotation):
                        print(l)
                        if idx == len(annotation) - 1:
                            f.write(f"{l}\n")
                        else:
                            f.write(f"{l} ")



def split_from_df(
        df: pd.DataFrame,
        class_col,
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
    classes = df[class_col].drop_duplicates()
    class_index = {i: n for i, n in enumerate(classes)}
    class_to_index = {n: i for i, n in class_index.items()}

    images = dict(df.groupby(class_col)['image'].apply(list))

    splits = {}
    for c, image_list in images.items():
        c = str(c)
        train, test_val = train_test_split(image_list, train_size=train_size, random_state=SEED)
        val, test = train_test_split(test_val, train_size=train_size, random_state=SEED)

        splits[c] = {'train': train, 'val': val, 'test': test}

        save_class_images(splits, c, df, class_to_index)


    save_yaml_file(DATASETS_FOLDER, class_index)
    return splits
