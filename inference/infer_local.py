import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from sahi.prediction import ObjectPrediction
from skimage.morphology import remove_small_objects
from skimage.measure import label

from dataset_generation.utils import convert_coco_to_yolo
from .sahi_segmentation import predict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dataset generation')
    parser.add_argument('--top-dir', type=str, default='/pool1/srv')
    parser.add_argument('--input-file', type=str, help='the subpath to the input file')
    parser.add_argument('--output-file', type=str, help='the name of the output file')
    parser.add_argument('--output-dir', type=str, default='cvat-tasks/infer-local',
                        help='the subpath dir to put the output file')
    parser.add_argument('--save-img', action='store_true', help='save the inference img example locally')
    parser.add_argument('--apply-bridge-splitting', action='store_true', help='apply morphology bridge splitting')
    parser.add_argument('--open-radius-px', type=int, default=1, help='number of pixels for morphology opening, disk')
    parser.add_argument('--anno-size-gte', type=int, default=50)
    args = parser.parse_args()
    return args


def label_original_components(cropped_mask):
    labeled = label(cropped_mask, connectivity=1)
    num_labels = labeled.max()
    return labeled, num_labels


def remove_small_attachments(labeled_orig, min_size_px: int):
    cleaned_labeled = remove_small_objects(labeled_orig, min_size=min_size_px, connectivity=1)
    cleaned_mask = cleaned_labeled > 0
    return cleaned_mask


def object_prediction_to_bbox_mask_local(obj):
    """
    Retrieve the bbox-local mask directly from SAHI's Mask object.
    """
    if hasattr(obj.mask, "bool_mask") and obj.mask.bool_mask is not None:
        return obj.mask.bool_mask.astype(bool)

    print('Warning: bool_mask not found, returning full_shape crop')
    # Fallback to full_shape crop if bool_mask isn't directly available
    full_mask = obj.mask.to_bool_mask()
    xmin = int(round(float(obj.bbox.minx)))
    ymin = int(round(float(obj.bbox.miny)))
    xmax = int(round(float(obj.bbox.maxx)))
    ymax = int(round(float(obj.bbox.maxy)))
    return full_mask[ymin:ymax, xmin:xmax]


def label_cleaned_components(cleaned_mask):
    labeled = label(cleaned_mask, connectivity=1)
    num_labels = labeled.max()
    return labeled, num_labels


def cleaned_mask_to_segments_cv2(cleaned_mask):
    """
    Extract clean, ordered exterior contours using OpenCV.
    Returns a list of flattened COCO-style coordinate lists [[x1, y1, x2, y2, ...]].
    """
    mask_uint8 = (cleaned_mask.astype(np.uint8)) * 255
    # cv2.RETR_EXTERNAL extracts outer boundaries only (ignores inner holes)
    # cv2.CHAIN_APPROX_SIMPLE removes redundant points on straight lines
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        contour = contour.squeeze(axis=1)  # Shape (N, 2) -> [x, y]
        if len(contour) < 3:
            continue

        segments.append(contour.ravel().tolist())
    return segments


def segments_to_bbox(segments):
    all_xy = []
    for seg in segments:
        coords = np.array(seg, dtype=float).reshape(-1, 2)
        all_xy.append(coords)
    if not all_xy:
        return None
    all_xy = np.vstack(all_xy)
    xs = all_xy[:, 0]
    ys = all_xy[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def split_cleaned_components_to_objects(obj, cleaned_mask):

    labeled_cleaned, num_labels_cleaned = label_cleaned_components(cleaned_mask)
    print("num_labels_cleaned:", num_labels_cleaned)

    new_objects = []

    for k in range(1, num_labels_cleaned + 1):
        component_mask = (labeled_cleaned == k)
        if component_mask.sum() == 0:
            continue

        # Extract exterior contours for this component
        segments = cleaned_mask_to_segments_cv2(component_mask)

        # Create an individual ObjectPrediction per continuous segment to avoid multi-polygon jumps
        for seg in segments:
            bbox = segments_to_bbox([seg])
            if not bbox:
                continue

            new_obj = ObjectPrediction(
                bbox=bbox,
                category_id=obj.category.id,
                category_name=obj.category.name,
                score=float(obj.score.value),
                segmentation=[seg],  # Exactly one outer polygon loop
                shift_amount=[0, 0],
                full_shape=[obj.mask.full_shape_height, obj.mask.full_shape_width],
            )
            new_objects.append(new_obj)

    return new_objects or [obj]


def split_self_bridges_remove_small(obj, min_size_px: int):
    has_mask = getattr(obj, "mask", None) is not None
    has_bbox = getattr(obj, "bbox", None) is not None
    if not has_mask or not has_bbox:
        print('WARNING: no mask or bbox')
        return [obj]

    cropped_mask = object_prediction_to_bbox_mask_local(obj)
    labeled_orig, num_labels = label_original_components(cropped_mask)
    cleaned_mask = remove_small_attachments(labeled_orig, min_size_px)
    print("num_labels:", num_labels)
    print("original sum:", cropped_mask.sum(), "cleaned sum:", cleaned_mask.sum())

    return split_cleaned_components_to_objects(obj, cleaned_mask)


def apply_bridge_splitting(pred_result, min_size_px: int):
    new_object_predictions = []
    for obj in pred_result.object_prediction_list:
        split_objs = split_self_bridges_remove_small(obj, min_size_px=min_size_px)
        new_object_predictions.extend(split_objs)
    pred_result.object_prediction_list = new_object_predictions
    return pred_result


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

    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    main(args)
