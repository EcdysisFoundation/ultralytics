import os
import random
import logging
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from .utils import save_yaml_file, check_minimum_length, VALID_IMG_EXTENSIONS


SEED = 42
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DATASETS_FOLDER = 'datasets'
DATASET_PANO = 'dataset_pano'
FULL_RESIZE_DIR = 'full_resized'


def create_clear_dirs(dataset_pano=None):
    parent_images = Path(DATASETS_FOLDER) / 'images'
    parent_labels = Path(DATASETS_FOLDER) / 'labels'

    # Clear previous runs, make fresh directories
    if os.path.exists(parent_images):
        shutil.rmtree(parent_images)
    if os.path.exists(parent_labels):
        shutil.rmtree(parent_labels)
    if dataset_pano:
        dp_path = Path(DATASET_PANO)
        if os.path.exists(dp_path):
            shutil.rmtree(dp_path)
        dp_path.mkdir()
        full_resized = Path(f'{DATASET_PANO}/{FULL_RESIZE_DIR}')
        full_resized.mkdir()

    subfolders = ('train', 'val', 'test')
    for name in subfolders:
        i = parent_images / name
        i.mkdir(parents=True)
        ld = parent_labels / name
        ld.mkdir(parents=True)

    return {
        'parent_images': parent_images,
        'parent_labels': parent_labels
    }


def create_clear_dirs_eval(eval_dataset_dir):
    images_path = Path(eval_dataset_dir) / 'images' / 'test'
    labels_path = Path(eval_dataset_dir) / 'labels' / 'test'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    return {
        'dataset_dir': eval_dataset_dir,
        'images_path': images_path,
        'labels_path': labels_path
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
            shutil.copy(src, dst, follow_symlinks=True)
        except shutil.SameFileError:
            logger.warning(f'File {dst} already present, skipping')

    for split_name, split_img in splits[c].items():
        if len(split_img) == 0:
            continue

        parent_i = dirs['parent_images'] / split_name
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

    df = df.copy()

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
        train, val, test = train_test_split(image_list, train_size=train_size, random_state=SEED)

        splits[c] = {'train': train, 'val': val, 'test': test}

        save_class_images(splits, c, df, class_to_index, dirs, args)

    save_yaml_file(DATASETS_FOLDER, class_index)
    return splits


def split_by_labels_train_val(label_dir, image_dir, base_dirs, itestset):
    """
    Using a directory of labelfiles and imgs, structure traing set for one class.
    Supports only one, defined img_ext at a time.
    """
    print('Starting split_by_labels_train_val')

    label_path = Path(label_dir)
    img_path = Path(image_dir)
    img_val = base_dirs['parent_images'] / 'val'
    img_train = base_dirs['parent_images'] / 'train'
    label_val = base_dirs['parent_labels'] / 'val'
    label_train = base_dirs['parent_labels'] / 'train'

    if itestset:
        img_test = base_dirs['parent_images'] / 'test'
        label_test = base_dirs['parent_labels'] / 'test'

    def copy_imgs(entries, img_set_path, label_set_path):
        copied_entries = 0
        for img_e in entries:
            if Path(img_e).suffix.lower() in VALID_IMG_EXTENSIONS:
                label_file = Path(img_e).with_suffix('.txt')
                full_label_path = label_path / label_file
                # Images may be present that do not have annotations because they were empty
                # and no yolo version files were made in that case. We check for that here.
                if full_label_path.exists():
                    try:
                        shutil.copy(img_path / img_e, img_set_path)
                        shutil.copy(full_label_path, label_set_path)
                        copied_entries += 1
                    except Exception as e:
                        print(f'Warning: {img_e} and {label_file} will not be included in training: {e}')
        return copied_entries

    all_entries = os.listdir(image_dir)
    num_in_validation = int(len(all_entries) * 0.2)
    random_entries = random.sample(all_entries, num_in_validation)
    if itestset:
        num_half_random = int(len(random_entries) * 0.5)
        val_entries = random.sample(random_entries, num_half_random)
        test_entries = [v for v in random_entries if v not in val_entries]
        copied_test_entries = copy_imgs(test_entries, img_test, label_test)
        print(f'copied {copied_test_entries} test_entries')
    else:
        val_entries = random_entries
    train_entries = [v for v in all_entries if v not in val_entries]
    copied_entries_val = copy_imgs(val_entries, img_val, label_val)
    copied_entries_train = copy_imgs(train_entries, img_train, label_train)
    print(f'copied {copied_entries_val} val_entries')
    print(f'copied {copied_entries_train} train_entries')
    print('Completed split_by_labels_train_val.')
