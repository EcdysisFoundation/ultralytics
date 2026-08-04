import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage.draw import polygon2mask
from skimage.morphology import opening, disk, remove_small_objects
from skimage.measure import label, find_contours

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
                        help='the name of the output file')
    parser.add_argument('--output-dir', type=str,
                        default='cvat-tasks/infer-local',
                        help='the subpath dir to put the output file')
    parser.add_argument('--save-img', action='store_true',
                        help='save the inference img example locally')
    parser.add_argument('--apply-bridge-splitting', action='store_true',
                        help='apply morphology bridge splitting')
    parser.add_argument('--open-radius-px', type=int,
                        default=1,
                        help='number of pixels for morphology opening, disk')
    parser.add_argument('--anno-size-gte', type=int, default=50)
    args = parser.parse_args()
    return args


def predict(img_path, save_img_file=False):
    print(f'running prediction on device {DETECTION_MODEL.device}')
    result = get_sliced_prediction(
        img_path,
        DETECTION_MODEL,
        slice_height=2000,
        slice_width=2000,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
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


def label_original_components(cropped_mask):
    labeled = label(cropped_mask, connectivity=1)
    num_labels = labeled.max()
    return labeled, num_labels


def remove_small_attachments(labeled_orig, min_size_px: int):
    # This removes labels with area < min_size_px
    cleaned_labeled = remove_small_objects(labeled_orig, min_size=min_size_px, connectivity=1)
    cleaned_mask = cleaned_labeled > 0  # back to bool
    return cleaned_mask


def object_prediction_to_bbox_mask_local(obj):
    """
    Convert a SAHI ObjectPrediction with a Mask.segmentation into a
    bbox-cropped binary mask (numpy bool array).
    """

    mask_obj = obj.mask

    # 1. Read bbox in full-image coords
    xmin = int(round(float(obj.bbox.minx)))
    ymin = int(round(float(obj.bbox.miny)))
    xmax = int(round(float(obj.bbox.maxx)))
    ymax = int(round(float(obj.bbox.maxy)))

    h = ymax - ymin
    w = xmax - xmin

    # 2. Initialize bbox-local mask
    cropped_mask = np.zeros((h, w), dtype=bool)

    # 3. Rasterize each polygon into bbox-local mask
    for seg_flat in mask_obj.segmentation:
        poly_rc = coco_flat_to_local_rowcol(seg_flat, xmin, ymin)
        poly_mask = polygon2mask((h, w), poly_rc)  # bool array
        cropped_mask |= poly_mask

    return cropped_mask


def coco_flat_to_local_rowcol(seg_flat, xmin, ymin):
    """
    seg_flat: [x1, y1, x2, y2, ...] in full-image pixel coords
    xmin, ymin: top-left of bbox in full-image coords

    returns: (N, 2) array of (row, col) in bbox-local coords
    """
    coords = np.array(seg_flat, dtype=float).reshape(-1, 2)  # (N, 2) [x, y]
    x_full = coords[:, 0]
    y_full = coords[:, 1]

    # shift into bbox-local coords
    x_local = x_full - xmin
    y_local = y_full - ymin

    # polygon2mask expects (row, col) = (y, x)
    poly_rc = np.column_stack((y_local, x_local))
    return poly_rc


def split_self_bridges_remove_small(obj, min_size_px: int):
    has_mask = getattr(obj, "mask", None) is not None
    has_bbox = getattr(obj, "bbox", None) is not None
    if not has_mask or not has_bbox:
        return [obj]

    # 1. Build cropped_mask
    cropped_mask = object_prediction_to_bbox_mask_local(obj)

    # 2. Label original mask
    labeled_orig, num_labels = label_original_components(cropped_mask)
    if num_labels <= 1:
        return [obj]

    # 3. Remove small attachments
    cleaned_mask = remove_small_attachments(labeled_orig, min_size_px)

    print("num_labels:", num_labels)
    print("original sum:", cropped_mask.sum(), "cleaned sum:", cleaned_mask.sum())

    # 4. For now, keep original object prediction as-is.
    # You have cleaned_mask if you later want to rebuild polygons, but we *don’t* use it yet
    # so the insect’s polygon remains untouched.
    return [obj]


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
        f'File not found: {input_file}'
        return
    if not os.path.exists(output_file):
        Path(output_file).touch()

    print(f'performing inference on {input_file}')
    pred_result = predict(input_file, save_img_file=args.save_img)

    print(f'len(pred_result.object_prediction_list) before: {len(pred_result.object_prediction_list)}')
    pred_result = apply_bridge_splitting(pred_result, min_size_px=625)
    print(f'len(pred_result.object_prediction_list) after: {len(pred_result.object_prediction_list)}')


if __name__ == '__main__':

    args = get_args()

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    main(args)

"""
example:

python -m inference.infer_local \
--input-file label-studio/mydata/stitchermedia/0c5dc6cf-3d75-4434-ba11-a98736489b25/panorama.jpg \
--output-file 4124_sw_T2__0c5dc6cf-3d75-4434-ba11-a98736489b25__panorama.txt \
--output-dir cvat-tasks/4124_sw_T2_wopening \
--apply-bridge-splitting

"""
