import os
import argparse
from pathlib import Path

from dataset_generation.utils import convert_coco_to_yolo
from .sahi_segmentation import predict
from .utils import apply_bridge_splitting


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--top-dir', type=str, default='/pool1/srv')
    parser.add_argument('--input-file', type=str, help='the subpath to the input file')
    parser.add_argument('--output-file', type=str, help='the name of the output file')
    parser.add_argument('--output-dir', type=str, default='cvat-tasks/infer-local',
                        help='the subpath dir to put the output file')
    parser.add_argument('--save-img', action='store_true', help='save the inference img example locally')
    parser.add_argument('--apply-bridge-splitting', action='store_true', help='apply morphology bridge splitting')
    args = parser.parse_args()
    return args


def main(args):
    input_file = f'{args.top_dir}/{args.input_file}'
    output_file = f'{args.top_dir}/{args.output_dir}/{args.output_file}'
    if not os.path.exists(input_file):
        print(f'File not found: {input_file}')
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(output_file):
        Path(output_file).touch()

    print(f'performing inference on {input_file}')
    pred_result = predict(input_file, save_img_file=args.save_img)

    if args.apply_bridge_splitting:
        initial_len_result = len(pred_result.object_prediction_list)
        pred_result = apply_bridge_splitting(pred_result, min_size_px=1250)
        print(f'len(pred_result.object_prediction_list) before: {initial_len_result}')
        print(f'len(pred_result.object_prediction_list) after: {len(pred_result.object_prediction_list)}')

    original_width = pred_result.image_width
    original_height = pred_result.image_height
    coco_result = pred_result.to_coco_predictions(
        image_id=os.path.basename(input_file))

    yolo_annotations = convert_coco_to_yolo(coco_result, original_width, original_height)

    with open(output_file, mode="w", encoding="utf-8") as file:
        for i, cat in enumerate(yolo_annotations['classificaions']):
            polygon = yolo_annotations['segments'][i]
            if polygon:
                file.write(f"{cat} {' '.join(str(v) for v in polygon)}\n")


if __name__ == '__main__':
    args = get_args()
    main(args)
