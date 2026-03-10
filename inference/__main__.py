import json
import os
import torch
from pathlib import Path

from .dataset import get_stitcher_data
from ..dataset_generation.utils import convert_coco_to_yolo
from .sahi_segmentation import predict
from .utils import put_predictions, save_labeling_img


STITCHER_URL = 'http://ecdysis01.local:8090/'


def main():
    """
    SAHI inference
    Configurable for multiple use cases.
    Currently we create pre_annotations for CVAT.AI with these hardcoded settigs.
    """
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(torch.cuda.get_device_name(0))

    file_mount = '/pool1/srv/label-studio/mydata/stitchermedia'
    label_studio_img_dir = '/pool1/srv/label-studio/mydata/labeling_files'
    yolo_format_file_dir = '/pool1/srv/cvat-tasks'
    api_post_url = STITCHER_URL + 'update-predictions-coco/'
    anno_size_gte = 50  # limits minimum annotation bbox size
    save_predictions_to_db = True
    skip_if_annotations = True
    save_label_studio_files = False
    save_yolo_format_files = True

    dont_overwrite = False

    cvat_task_name = 'mytask'
    send_these_sites = []  # send based on sitecode example [str(i) for i in range(4111, 4131)]
    send_these_panos = []  # use the upload_dir, example [4308_sw_T2, ...]

    all_filters = send_these_sites + send_these_panos
    for site_or_dir in all_filters:
        filtered_data = get_stitcher_data(STITCHER_URL, site_or_dir)

        for d in filtered_data:
            # we use a name convention in first for characters, filter those
            if d['upload_dir_name'][:4] not in send_these_sites \
                    and d['upload_dir_name'] not in send_these_panos:
                continue
            if d['panorama_path']:
                if dont_overwrite and d['predictions_coco']:
                    print(f"dont_overwrite enabled, skipping {d['upload_dir_name']} has predictions")
                    continue
                if skip_if_annotations and d['annotations_segment']:
                    print(f"skip_if_annotations enabled, skipping {d['upload_dir_name']} has annotations")
                    continue
                p = file_mount + d['panorama_path']
                p = p.replace('/media', '')
                if os.path.exists(p):
                    print(f'performing inference on {p}')
                    coco_result, original_width, original_height = predict(p)
                    # filter missing bbox
                    coco_result = [v for v in coco_result if v['bbox']]
                    # filter based on bbox size
                    coco_result = [
                        v for v in coco_result if v['bbox'][2] >= anno_size_gte or v['bbox'][3] >= anno_size_gte
                    ]
                    if save_predictions_to_db:
                        prediction_result = json.dumps([{
                            'predictions': coco_result,
                            'original_width': original_width,
                            'original_height': original_height
                        }])
                        put_predictions(
                            api_post_url,
                            d['guid'],
                            prediction_result)

                    # usually only doing one or the other save_label_studio_files or save_yolo_format_files
                    if save_label_studio_files:
                        img_filename = str(Path(p).name)
                        new_filename_path = Path(f"{d['guid']}__{img_filename}")
                        full_img_save_path = str(Path(label_studio_img_dir) / new_filename_path)
                        save_labeling_img(p, full_img_save_path)

                    if save_yolo_format_files:
                        img_filename = str(Path(p).name)
                        new_filename_path = Path(f"{d['guid']}__{img_filename}")
                        full_img_save_path = str(Path(yolo_format_file_dir) / cvat_task_name / 'images' / 'train' / new_filename_path)
                        full_label_save_path = str(Path(yolo_format_file_dir) / cvat_task_name / 'labels' / 'train' / f'{new_filename_path}.txt')
                        save_labeling_img(p, full_img_save_path)
                        yolo_annotations = convert_coco_to_yolo(coco_result, original_width, original_height)
                        # write segmentation label file
                        with open(full_label_save_path, encoding="utf-8" ) as file:
                            for i, cat in enumerate(yolo_annotations['classificaions']):
                                polygon = yolo_annotations['segments'][i]
                                if polygon:
                                    file.write(f'{cat} {' '.join(str(v) for v in polygon)}\n')

                else:
                    print('path not found')
                    print(p)


if __name__ == '__main__':
    main()
