import argparse
import logging
from pathlib import Path
from .split import split_from_df, DATASETS_FOLDER
from .data import ObjectDetectData
from .utils import convert_annotation_to_yolo, check_missing_files, generate_split_class_report

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument(
        '--class-col', type=str, default='specimen__classification__gbif_order',
        help='The column to catagorize the images')
    parser.add_argument('-t', '--test-flag', action='store_true')
    parser.add_argument('-cpy', '--copy-files', action='store_true')

    return parser.parse_args()


def main():
    args = get_args()

    db = ObjectDetectData()
    full_data = db.get_full_df()

    category_counts = full_data[args.class_col].value_counts()

    logger.info('category counts')
    logger.info(category_counts)

    check_ok = check_missing_files(full_data, args.test_flag)

    if check_ok:
        print(check_ok)
    else:
        print('exiting...........')
        return

    full_data = db.get_full_df()
    full_data['yolo_annotations'] = full_data['object_det_label'].apply(convert_annotation_to_yolo)
    full_data.to_csv('local_files/full_data.csv')

    splits = split_from_df(full_data, args)
    report_count_df = generate_split_class_report(splits, args.class_col)
    report_count_df.to_csv(Path(DATASETS_FOLDER) / 'dataset_report.csv', index=False)


    print('end of main')

# This gets executed when running `python -m dataset_generation`
if __name__ == '__main__':
    main()
