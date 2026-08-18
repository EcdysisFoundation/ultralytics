import argparse
import os
import torch
from pathlib import Path

from .dataset import get_stitcher_data
from dataset_generation.utils import convert_coco_to_yolo
from .sahi_segmentation import predict
from .utils import apply_bridge_splitting, save_labeling_img


STITCHER_URL = 'http://ecdysis01.local:8090/'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--task-dir', type=str, help='sub directory for yolo_format_dir')
    parser.add_argument('--yolo_format_dir', type=str, default='/pool1/srv/cvat-tasks')
    parser.add_argument('--bypass-bridge-splitting', action='store_true', help='skip morphology bridge splitting')
    parser.add_argument('--skip-in-labeling-project', action='store_true',
                        help='do not use if in a labeling project')
    parser.add_argument('--anno-size-gte', type=int, default=50)
    parser.add_argument('--min-size-px', type=int, default=1250)
    parser.add_argument('--label-format', choices=['yolo', 'label_studio', 'skip'], default='yolo')
    parser.add_argument('--file-mount', type=str, default='/pool1/srv/label-studio/mydata/stitchermedia')
    parser.add_argument('--label-studio-img-dir', type=str, default='/pool1/srv/label-studio/mydata/labeling_files')
    parser.add_argument(
        '--site-range',
        type=lambda s: [item.strip() for item in s.split(',')],
        help='length 2, Comma-separated range of starting and ending site numbers'
    )
    parser.add_argument(
            '--panos',
            type=lambda s: [item.strip() for item in s.split(',')],
            help='Comma-separated pano site names'
        )
    args = parser.parse_args()
    if args.label_format == 'yolo' and not args.task_dir:
        parser.error("--task-dir is required when --label-format is 'yolo'")
    if args.site_range:
        if len(args.site_range) != 2:
            parser.error(f'--site-range requires length of 2, starting and ending. You entered {args.site_range}')
        v = [int(site) for site in args.site_range]
        args.site_range = v
    return args


def main(args):
    """
    SAHI inference
    Configurable for multiple use cases by hardcoded settings.
    """
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(torch.cuda.get_device_name(0))

    send_these_sites = [str(i) for i in range(args.site_range[0], args.site_range[1] + 1)]
    send_these_panos = args.panos if args.panos else []

    all_filters = send_these_sites + send_these_panos
    if len(all_filters) == 0:
        print('WARNING: no filter for sites or panos is not allowed, exiting')
        return
    for site_or_dir in all_filters:
        filtered_data = get_stitcher_data(STITCHER_URL, site_or_dir)

        for d in filtered_data:
            # we use a name convention in first for characters, filter those
            if d['upload_dir_name'][:4] not in send_these_sites \
                    and d['upload_dir_name'] not in send_these_panos:
                continue
            if d['label_studio_project'] and args.skip_in_labeling_project:
                print(f"--skip-in-labeling-project enabled, skipping {d['upload_dir_name']}")
                continue
            if d['panorama_path']:
                img_path = args.file_mount + d['panorama_path']
                img_path = img_path.replace('/media', '')
                if os.path.exists(img_path):
                    print(f'performing inference on {img_path}')
                    print(f"upload_dir_name is {d['upload_dir_name']}")
                    pred_result = predict(img_path)
                    original_width = pred_result.image_width
                    original_height = pred_result.image_height
                    if not args.bypass_bridge_splitting:
                        initial_len_result = len(pred_result.object_prediction_list)
                        pred_result = apply_bridge_splitting(pred_result, min_size_px=args.min_size_px)
                        print(f'len(pred_result.object_prediction_list) before: {initial_len_result}')
                        len_result = len(pred_result.object_prediction_list)
                        print(f'len(pred_result.object_prediction_list) after: {len_result}')
                    coco_result = pred_result.to_coco_predictions(image_id=os.path.basename(img_path))
                    if args.bypass_bridge_splitting:
                        # filter data without skimage
                        # remove missing bbox
                        coco_result = [v for v in coco_result if v['bbox']]
                        # filter based on bbox size
                        coco_result = [
                            v for v in coco_result if v['bbox'][2] >=
                            args.anno_size_gte or v['bbox'][3] >= args.anno_size_gte
                        ]

                    if args.label_format == 'label_studio':
                        img_filename = str(Path(img_path).name)
                        new_filename_path = Path(f"{d['guid']}__{img_filename}")
                        full_img_save_path = str(Path(args.label_studio_img_dir) / new_filename_path)
                        save_labeling_img(img_path, full_img_save_path)

                    elif args.label_format == 'yolo':
                        img_filename = str(Path(img_path).name)
                        new_filename_path = Path(f"{d['upload_dir_name']}__{d['guid']}__{img_filename}")
                        task_name_path = Path(args.yolo_format_dir) / args.task_dir
                        task_name_path.mkdir(parents=True, exist_ok=True)
                        full_img_save_path = str(task_name_path / new_filename_path)
                        full_label_save_path = str(task_name_path / f'{new_filename_path.stem}.txt')
                        save_labeling_img(img_path, full_img_save_path)
                        yolo_annotations = convert_coco_to_yolo(coco_result, original_width, original_height)
                        # write segmentation label file
                        with open(full_label_save_path, mode="w", encoding="utf-8") as file:
                            for i, cat in enumerate(yolo_annotations['classificaions']):
                                polygon = yolo_annotations['segments'][i]
                                if polygon:
                                    file.write(f"{cat} {' '.join(str(v) for v in polygon)}\n")

                else:
                    print('path not found')
                    print(img_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
