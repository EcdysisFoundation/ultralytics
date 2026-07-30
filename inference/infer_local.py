import os
import argparse
from pathlib import Path
from PIL import Image

from sahi.predict import get_sliced_prediction

from dataset_generation.utils import convert_coco_to_yolo
from .sahi_segmentation import DETECTION_MODEL


def get_args() -> argparse.Namespace:
    # python -m dataset_generation
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--top-dir', type=str,
                        default='/pool1/srv')
    parser.add_argument('--input-file', type=str,
                        help='the subpath to the input file')
    parser.add_argument('--output-file', type=str,
                        help='the subpath to put the output file')
    parser.add_argument('--save-img', action='store_true',
                        help='save the inference img example locally')
    parser.add_argument('--anno-size-gte', type=int, default=50)
    args = parser.parse_args()
    return args


def predict(img_path, save_img_file=False):
    print(f'running prediction on device {DETECTION_MODEL.device}')
    result = get_sliced_prediction(
        img_path,
        DETECTION_MODEL,
        slice_height=6000,
        slice_width=6000,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_threshold=0.4,
        perform_standard_pred=True,
        postprocess_match_metric='IOS',  # default IOS
        postprocess_type="GREEDYNMM"  # default GREEDYNMM
    )
    # optionally save image file
    if save_img_file:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        result.export_visuals(
            export_dir="local_files/output/inference",
            file_name=filename,
            hide_labels=True,
            hide_conf=True)

    return result


def main(args):
    input_file = f'{args.top_dir}/{args.input_file}'
    output_file = f'{args.top_dir}/{args.output_file}'
    if not os.path.exists(input_file):
        f'File not found: {input_file}'
        return
    if not os.path.exists(output_file):
        Path(output_file).touch()

    print(f'performing inference on {input_file}')
    pred_result = predict(input_file, save_img_file=args.save_img)

    # examine object_prediction_list
    print(coco_result.object_prediction_list[0].__dict__)

    for obj in coco_result.object_prediction_list:
        # masks: often something like obj.mask or obj.mask_numpy
        has_mask = getattr(obj, "mask", None) is not None

        # polygons: SAHI typically stores segmentation as obj.segment or obj.contours
        has_polygon = getattr(obj, "segment", None) is not None or \
            getattr(obj, "polygon", None) is not None

        print(obj.category_name, has_mask, has_polygon)

    original_width = pred_result.image_width
    original_height = pred_result.image_height
    coco_result = pred_result.to_coco_predictions(
            image_id=os.path.basename(input_file))

    # filters
    print(f'{len(coco_result)} annotations before filtering')
    # filter missing bbox
    coco_result = [v for v in coco_result if v['bbox']]
    print(f'{len(coco_result)} after filtering missing box')
    # filter based on bbox size
    coco_result = [
        v for v in coco_result if v['bbox'][2] >= args.anno_size_gte or v['bbox'][3] >= args.anno_size_gte
    ]
    print(f'{len(coco_result)} after filtering small boxes')

    yolo_annotations = convert_coco_to_yolo(coco_result, original_width, original_height)
    # write segmentation label file
    with open(output_file, mode="w", encoding="utf-8") as file:
        for i, cat in enumerate(yolo_annotations['classificaions']):
            polygon = yolo_annotations['segments'][i]
            if polygon:
                file.write(f"{cat} {' '.join(str(v) for v in polygon)}\n")


if __name__ == '__main__':
    args = get_args()

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    main(args)
