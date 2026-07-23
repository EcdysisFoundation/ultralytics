import copy
import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from ultralytics.data.converter import convert_coco

from PIL import Image

from .split import (
    create_clear_dirs, create_clear_dirs_eval, split_from_df,
    split_by_labels_train_val, DATASETS_FOLDER,
    DATASET_PANO, FULL_RESIZE_DIR
)
from .stitcher_api import (
    pano_segmentation_training_set_fromyolo, count_initial_training_recs,
    get_root_message, ERROR_MSG_KEY
)
from .eval_test_dataset import eval_test_dataset
from .data import ObjectDetectData
from .utils import (
    convert_annotation_to_yolo, check_missing_files, generate_split_class_report,
    resize_imgs_in_dir, filter_small_yolo_annotations, remove_without_annotations
)

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)

DATASET_JSON = 'dataset.json'
COCO_JSON_SOURCE = {
        "images": [],
        "categories": [{
            "id": 1,
            "name": "arthropod"}],
        "annotations": [],
    }

PLATFORM_CVAT = 'cvat'
PLATFORM_LABEL_STUDIO = 'label-studio'


def get_args() -> argparse.Namespace:
    # python -m dataset_generation
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument(
        '--class-col', type=str, default='specimen__classification__gbif_order',
        help='The column to catagorize the images')
    parser.add_argument('-t', '--test-flag', action='store_true')
    parser.add_argument('-c', '--count-only', action='store_true')
    parser.add_argument('-e', '--evaluation-only', action='store_true')
    parser.add_argument('-local', '--local-train-dataset', action='store_true',
                        help=("when set to true, this uses files already in DATASET_PANO and "
                              "does not call api to get new ones. Evaluation dataset does not change"))
    parser.add_argument('--omit-fullsize', action='store_true',
                        help="dont include the full size images to train for setting perform_standard_pred=True")
    parser.add_argument('--eval-percent', type=int, default=10,
                        help="percent of images to be placed in evaluation set")
    parser.add_argument('--slice-size', type=int, default=2000,
                        help="image slicing dimensions for SAHI")
    parser.add_argument('--anno-size-gte', type=int, default=50)
    parser.add_argument('-itestset', '--include-testset', action='store_true')
    parser.add_argument(
        '--label-platform',
        choices=[PLATFORM_CVAT, PLATFORM_LABEL_STUDIO],
        default=PLATFORM_CVAT)
    args = parser.parse_args()
    if args.eval_percent < 0 or args.eval_percent > 100:
        raise ValueError(f'args.eval_percent cannot be < 0 or > 100, you entered {args.eval_percent}')
    if args.evaluation_only and args.local_train_dataset:
        raise ValueError('args.evaluation_only and args.local_train_dataset at the same time is not valid.')
    return args


def single_specimen_trainingset(check_missing=True):
    args = get_args()

    db = ObjectDetectData()
    full_data = db.get_full_df()
    category_counts = full_data[args.class_col].value_counts()
    logger.info('category counts')
    logger.info(category_counts)

    if check_missing:
        check_ok = check_missing_files(full_data, args.test_flag)
        if check_ok:
            print(check_ok)
        else:
            print('exiting...........')
            return

    full_data['yolo_annotations'] = full_data['object_det_label'].apply(convert_annotation_to_yolo)
    full_data.to_csv('local_files/full_data.csv')

    splits = split_from_df(full_data, args)
    report_count_df = generate_split_class_report(splits, args.class_col)
    report_count_df.to_csv(Path(DATASETS_FOLDER) / 'dataset_report.csv', index=False)

    print('end of main')


def slice_pano_training_set(
        args,
        dataset_dir,
        dataset_json_dir,
        dataset_sliced_dir):

    print(f'dataset_file_path: {dataset_json_dir}')

    coco_dict = load_json(dataset_json_dir)
    len_images = len(coco_dict['images'])
    if not len_images:
        print('There are no images in the training dataset.')
    print(f'There are {len_images} images in the dataset')
    print(f"coco_dict read, first image is {coco_dict['images'][0]['file_name']}")

    slice_coco(
        coco_annotation_file_path=dataset_json_dir,
        image_dir=dataset_dir,
        output_coco_annotation_file_name="sliced_coco.json",
        ignore_negative_samples=False,
        output_dir=dataset_sliced_dir,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
        min_area_ratio=0.1,
        verbose=True
    )
    print('slice_pano_training_set done')


def main(args):

    if not args.local_train_dataset:
        api_ping = get_root_message()
        print(api_ping)
        if ERROR_MSG_KEY in api_ping.keys():
            print(api_ping)
            print('Stitcher-FastAPI is not reachable, exting...')
            return

    curr_dir = os.getcwd()
    coco_converted_prefix = 'coco_converted'  # this will get created as coco_converted-2 if exists
    coco_conv_dir = f'{DATASET_PANO}/{coco_converted_prefix}'
    dataset_json_dir = f'{curr_dir}/{DATASET_PANO}/{DATASET_JSON}'
    dataset_dir = f'{curr_dir}/{DATASET_PANO}'
    slice_dir = f'{DATASET_PANO}/sliced'
    dataset_sliced_dir = f'{curr_dir}/{slice_dir}'
    sliced_coco_json_dir = f'{coco_conv_dir}/labels/sliced_coco.json_coco'  # this is a directory
    eval_dataset_dir = f'{curr_dir}/eval_dataset_pano'

    if args.evaluation_only:
        print('args.evaluation_only is set to True, previous training data will not be cleared')
    else:
        print('clearing previous training data....')
        if args.local_train_dataset:
            # clear training dataset directories, not DATASET_PANO
            base_dirs = create_clear_dirs()
            # however, we need to clear any coco_converted, coco_converted-2, etc.
            parent_dir = Path(DATASET_PANO)
            for item_path in parent_dir.iterdir():
                if item_path.is_dir() and item_path.name.startswith(coco_converted_prefix):
                    shutil.rmtree(item_path)
        else:
            # clear DATASET_PANO to download new ones.
            print(f'clearing previous {DATASET_PANO} directory, will download new panos')
            base_dirs = create_clear_dirs(dataset_pano=True)

    if not args.local_train_dataset:
        if args.label_platform == PLATFORM_CVAT:
            print(f'deleting anything in {eval_dataset_dir} to start new')
            eval_dirs = create_clear_dirs_eval(eval_dataset_dir)
            pano_segmentation_training_set_fromyolo(
                dataset_dir,
                DATASET_JSON,
                copy.deepcopy(COCO_JSON_SOURCE),
                args,
                eval_dirs,
                FULL_RESIZE_DIR)
        else:
            print(f'--label-platform {args.label_platform} not supported')

        # make the evaluation test dataset of non-sliced images
        # these were placed in eval_dirs from pano_segmentation_training_set_fromyolo
        eval_test_dataset(eval_dirs)

    if not args.evaluation_only:
        slice_pano_training_set(
            args,
            dataset_dir,
            dataset_json_dir,
            dataset_sliced_dir)
        convert_coco(
            slice_dir,
            cls91to80=False,
            save_dir=coco_conv_dir,
            use_segments=True)
        split_by_labels_train_val(sliced_coco_json_dir, slice_dir, base_dirs, args.include_testset)
        if not args.omit_fullsize:
            print('including full images along with slices, use --omit-fullsize to not')
            resize_imgs_in_dir(dataset_dir, FULL_RESIZE_DIR, args.slice_size)
            resize_dir = f'{dataset_dir}/{FULL_RESIZE_DIR}'
            # use args.anno_size_gte to filter smallest annotations,
            # but these will be smallar if anno_size_gte * (imgsz setting / slice-size) < anno_size_gte
            filter_small_yolo_annotations(
                image_dir=resize_dir, label_dir=resize_dir, min_pixel_size=args.anno_size_gte)
            remaining_images = remove_without_annotations(image_dir=resize_dir, label_dir=resize_dir)
            if remaining_images >= 6:
                split_by_labels_train_val(resize_dir, resize_dir, base_dirs, args.include_testset)


# run with `python -m dataset_generation -t`
if __name__ == '__main__':
    """
    Assumes running from ultralytics home dir with 'python -m dataset_generation'
    """
    args = get_args()

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    initial_count = count_initial_training_recs()
    if initial_count and not args.count_only:
        main(args)
