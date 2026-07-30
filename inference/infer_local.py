import os
import argparse
from pathlib import Path
from PIL import Image

from dataset_generation.utils import convert_coco_to_yolo
from .sahi_segmentation import predict


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


def main(args):
    input_file = f'{args.top_dir}/{args.input_file}'
    output_file = f'{args.top_dir}/{args.output_file}'
    if not os.path.exists(input_file):
        f'File not found: {input_file}'
        return
    if not os.path.exists(output_file):
        Path(output_file).touch()

    print(f'performing inference on {input_file}')
    coco_result, original_width, original_height = predict(
        input_file, save_img_file=args.save_img)

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
