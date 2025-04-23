import argparse
import logging

from .split import split_from_df
from .data import ObjectDetectData
from .utils import convert_annotation_to_yolo, check_missing_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument(
        '--class-name', type=str, default='specimen__classification__gbif_order',
        help='myhelp')

    return parser.parse_args()


def main():
    args = get_args()

    db = ObjectDetectData()
    data = db.get_image_df()

    check_ok = check_missing_files(data)

    if check_ok:
        print(check_ok)
    else:
        print('exiting...........')
        return

    full_data = db.get_full_df()
    full_data['yolo_annotations'] = full_data['object_det_label'].apply(convert_annotation_to_yolo)
    full_data.to_csv('local_files/full_data.csv')

    split_from_df(full_data, args.class_name)

    print('end of main')

# This gets executed when running `python -m dataset_generation`
if __name__ == '__main__':
    main()
