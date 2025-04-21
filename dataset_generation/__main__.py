import argparse
import pandas as pd
from pathlib import Path

from .data import ObjectDetectData
from .utils import convert_annotation_to_yolo, check_missing_files


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Dataset generation')
    #parser.add_argument('--my-arg', type=str, default='mydefault',
    #                    help='myhelp')

    return parser.parse_args()


def split_from_df(df: pd.DataFrame, output: Path, train_size=0.8, seed=42):

    return


def main():
    args = get_args()

    db = ObjectDetectData()
    data = db.get_image_df()

    check_ok = check_missing_files(data)

    #if check_ok:
    #    print(check_ok)
    #else:
    #    print('exiting...........')
    #    return
    full_data = db.get_full_df()
    full_data['yolo_annotations'] = full_data['object_det_label'].apply(convert_annotation_to_yolo)
    full_data.to_csv('local_files/full_data.csv')
    print(full_data[['object_det_label', 'yolo_annotations']])

# This gets executed when running `python -m dataset_generation`
if __name__ == '__main__':
    main()
