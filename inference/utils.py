import cv2
from math import sqrt
import numpy as np

from sahi.prediction import ObjectPrediction
from skimage.morphology import remove_small_objects
from skimage.measure import label


def save_labeling_img(path, full_img_save_path):
    """
    Resizes for labeling app, below Decompression Bomb threshold.
    Saves to labeling directory even if not resized.
    """
    try:
        img = cv2.imread(path)
    except cv2.error as e:
        print(f"Error loading image: {e}")
        img = None
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    max_pixels = 128_000_000  # Decompression Bomb warning threshold
    margin = 0.95
    target_pixels = margin * max_pixels
    h, w = img.shape[:2]
    pixels = w * h
    print(f'img pixels are {pixels}')
    if pixels > target_pixels:
        print(f'> target of {target_pixels} resizing {path}')
        scale = sqrt(target_pixels / pixels)
        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        success = cv2.imwrite(full_img_save_path, resized)
    else:
        success = cv2.imwrite(full_img_save_path, img)

    if not success:
        raise IOError(f"Could not write image: {full_img_save_path}")
    return


def label_original_components(cropped_mask):
    labeled = label(cropped_mask, connectivity=1)
    num_labels = labeled.max()
    return labeled, num_labels


def remove_small_attachments(labeled_orig, min_size_px: int):
    binary_input = labeled_orig > 0
    cleaned_mask = remove_small_objects(binary_input, min_size=min_size_px, connectivity=1)
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


def split_cleaned_components_to_objects(obj, cleaned_mask, num_labels):
    labeled_cleaned, num_labels_cleaned = label_cleaned_components(cleaned_mask)
    if num_labels != num_labels_cleaned:
        print(f'number_labels: {num_labels}, cleaned_number_labels: {num_labels_cleaned}')
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
    cropped_mask_sum = cropped_mask.sum()
    cleaned_mask_sum = cleaned_mask.sum()
    if cropped_mask_sum != cleaned_mask_sum:
        print(f'cropped_mask_sum: {cropped_mask_sum}, cleaned_mask_sum: {cleaned_mask_sum}')
    return split_cleaned_components_to_objects(obj, cleaned_mask, num_labels)


def apply_bridge_splitting(pred_result, min_size_px: int):
    new_object_predictions = []
    for obj in pred_result.object_prediction_list:
        split_objs = split_self_bridges_remove_small(obj, min_size_px=min_size_px)
        new_object_predictions.extend(split_objs)
    pred_result.object_prediction_list = new_object_predictions
    return pred_result
