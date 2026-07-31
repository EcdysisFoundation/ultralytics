import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon
from skimage.draw import polygon2mask
from skimage.morphology import opening, disk
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


def open_mask_break_bridges(cropped_mask, radius_px: int):
    """
    Apply morphological opening with a disk SE to break thin self-bridges.
    """
    selem = disk(radius_px)  # flat disk structuring element
    opened = opening(cropped_mask, selem)  # erosion then dilation
    return opened


def label_components(opened_mask):
    """
    Label connected components in a binary mask.
    Returns labeled image and number of labels.
    """
    labeled = label(opened_mask, connectivity=1)  # 4-connectivity in 2D
    num_labels = labeled.max()
    return labeled, num_labels


def component_label_to_polygons(labeled, k):
    """
    labeled: int array of labels
    k: component label (1..num_labels)
    returns: list of shapely Polygons in bbox-local coords
    """
    component_mask = (labeled == k)

    # find_contours expects float image; coords are (row, col)
    contours = find_contours(component_mask.astype(float), level=0.5)

    polys = []
    for contour in contours:
        # contour[:, 0] = row (y), contour[:, 1] = col (x)
        y = contour[:, 0]
        x = contour[:, 1]
        coords = np.column_stack((x, y))  # shapely expects (x, y)
        poly = Polygon(coords)
        if poly.area > 0:
            polys.append(poly)

    return polys


def split_self_bridges(obj):
    has_mask = getattr(obj, "mask", None) is not None
    has_bbox = getattr(obj, "bbox", None) is not None

    if not has_mask or not has_bbox:
        return [obj]  # nothing to see here

    cropped_mask = object_prediction_to_bbox_mask_local(obj)
    print('cropped_mask.__dir__')
    print(cropped_mask.__dir__)
    opened = open_mask_break_bridges(cropped_mask, 2)
    print("original sum:", cropped_mask.sum(), "opened sum:", opened.sum())
    labeled, num_labels = label_components(opened)
    if num_labels > 1:
        polys = component_label_to_polygons(labeled, num_labels)
        print(polys)
    print("num_labels:", num_labels)


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

    for i, pred in enumerate(pred_result.object_prediction_list):

        # pred == {'score': PredictionScore: <value: ...>, 'mask': <sahi.annotation.Mask object at ...>, 'bbox': BoundingBox: <(..., ..., ..., ...), w: ..., h: ...>, 'category': Category: <id: ..., name: item>, 'merged': None}

        # print(pred.mask.__dict__)
        # {'shift_x': 0, 'shift_y': 0, 'full_shape_height': ..., 'full_shape_width': ..., 'segmentation': [[..., ...

        split_self_bridges(pred)

        # temp stop early
        if i >= 2:
            break

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

"""
example:

python -m inference.infer_local \
--input-file label-studio/mydata/stitchermedia/0c5dc6cf-3d75-4434-ba11-a98736489b25/panorama.jpg \
--output-file cvat-tasks/texas_oklahoma_2025c_rerun/4124_sw_T2__0c5dc6cf-3d75-4434-ba11-a98736489b25__panorama.txt

"""
