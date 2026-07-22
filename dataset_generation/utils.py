import sys
import logging
import numpy as np
import os
import pandas as pd
import yaml
from PIL import Image

from pathlib import Path
from uuid import uuid4
from sahi.utils.coco import CocoAnnotation

from inference.sahi_stitched import label_studio_to_coco


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

logger.setLevel(logging.INFO)

FILE_MOUNT = '/pool1/srv/label-studio/mydata/stitchermedia'


def make_yaml_dict(dataset_folder, class_index):
    return {
            'path': '../' + dataset_folder,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': class_index
        }


def save_yaml_file(dataset_folder, class_names):
    yaml_name = 'data.yaml'
    y = make_yaml_dict(dataset_folder, class_names)
    with (Path(dataset_folder)/yaml_name).open('w') as f:
        yaml.dump(y, f)


def check_missing_files(data, test_flag):
    # check if files exist
    missingcsv = 'local_files/missing_images.csv'
    logger.info('Checking for missing images ...')
    data['exists'] = data['full_image_path'].astype(str).map(os.path.exists)
    missing_images = data[data['exists'] == False]
    if len(missing_images):
        v = len(missing_images)
        if v >= 20:
            v = 20
        logger.info('some images are missing. Up to the first 20 are...')
        logger.info(missing_images.iloc[0:v])
        logger.info('saving to file {0} ....'.format(missingcsv))
        missing_images.to_csv(missingcsv)
        if not test_flag:
            return None
        else:
            return 'In Testing mode, we found missing images...'
    return 'All images found'


def convert_annotation_to_yolo(labels):
    """
    modified from:
    https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/utils.py

    Convert LS Bounding Box annotation to Yolo format.

    Args:
        label (dict): Dictionary containing annotation information including:
            - width (float): Width of the object.
            - height (float): Height of the object.
            - x (float): X-coordinate of the top-left corner of the object.
            - y (float): Y-coordinate of the top-left corner of the object.

    Returns:
        tuple or None: If the conversion is successful, returns a tuple (x, y, w, h) representing
        the coordinates and dimensions of the object in Yolo format, where (x, y) are the center
        coordinates of the object, and (w, h) are the width and height of the object respectively.
    """
    result = []
    for label in labels:
        if ("x" in label and "y" in label and 'width' in label and 'height' in label):
            w = label['width']
            h = label['height']

            x = (label['x'] + w / 2) / 100
            y = (label['y'] + h / 2) / 100
            w = w / 100
            h = h / 100

            result.append([x, y, w, h])

    return result


def check_minimum_length(image_list, train_size):
    x = len(image_list) * train_size / 2
    if x >= 2.0:
        return True
    else:
        return False


def get_count_per_class_split(splits, class_name):
    """
    Get the number of images per class in each split
    splits has the following format (as in the splits.yaml file)
    {
     '99': {
            'test': [  '/path/to/test_image1_for_class_99.jpg',... ],
            'train': [  '/path/to/train_image1_for_class_99.jpg',...  ],
            'val': [  '/path/to/val_image1_for_class_99.jpg', ...]
        },
        ...
    }
    Args:
        splits: Dictionary of lists of image paths per split, the key is the class name, the value is a dict of split, list of image path of that split and class
    Returns:
        Dataframe with the number of images per class in each split, columns are split names (train,test,val), rows are class ids
    """
    counts = []

    for class_id, split in splits.items():
        # id, train, test, val
        counts.append({class_name: class_id, **{split_name: len(image_paths) for split_name, image_paths in split.items()}})
    return pd.DataFrame(counts)


def generate_split_class_report(splits, class_name):
    """
    Return the dataset sample count report
    """

    counts_df = get_count_per_class_split(splits, class_name)

    counts_df["total_samples"] = counts_df["train"] + counts_df["val"] + counts_df["test"]

    return counts_df.sort_values(by=class_name)


def normalize_polygons_flat_np(polygons, image_width, image_height):
    # polygons: list of flat lists, variable lengths allowed
    w = float(image_width)
    h = float(image_height)
    return [
        (np.asarray(poly, dtype=np.float32) / np.where(
            np.arange(len(poly)) % 2 == 0,  # x indices
            w,
            h
        )).tolist()
        for poly in polygons
    ]


def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple[int, int]): A tuple (idx1, idx2) where idx1 is the index in arr1 and idx2 is the index in arr2 of the
            pair with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """Merge multiple segments into one list by connecting the coordinates with the minimum distance between each
    segment.

    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like
            [segmentation1, segmentation2,...].

    Returns:
        (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def convert_coco_to_yolo(coco_result, image_width, image_height, use_keypoints=False):
    """
    Modified from convert_coco at
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py
    and requirements of
    https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/
    """
    boxes = []
    segments = []
    keypoints = []
    classificaions = []
    for anno in coco_result:
        if anno.get("iscrowd", False):
            continue
        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(anno["bbox"], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= image_width  # normalize x
        box[[1, 3]] /= image_height  # normalize y
        if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
            continue
        boxes.append(box)
        classificaions.append(anno['category_id'])

        if not anno.get("segmentation"):
            segments.append([])
        elif len(anno["segmentation"]) > 1:
            # sometimes multiple polygons are predicted for a single object
            s = merge_multi_segment(anno["segmentation"])
            s = (np.concatenate(s, axis=0) / np.array([image_width, image_height])).reshape(-1).tolist()
        else:
            s = [j for i in anno["segmentation"] for j in i]  # all segments concatenated
            s = (np.array(s).reshape(-1, 2) / np.array([image_width, image_height])).reshape(-1).tolist()
        segments.append(s)

        if use_keypoints:
            if anno.get("keypoints") is None:
                keypoints.append([])
            keypoints.append(
                box + (np.array(anno["keypoints"]).reshape(-1, 3) / np.array([image_width, image_height, 1])).reshape(-1).tolist()
            )
    assert len(boxes) == len(classificaions)
    if segments:
        assert len(boxes) == len(classificaions) == len(segments)
    if keypoints:
        assert len(boxes) == len(classificaions) == len(segments) == len(keypoints)

    return {
        'boxes': boxes,
        'classificaions': classificaions,
        'segments': segments,
        'keypoints': keypoints
    }


def get_polygon_area(x, y):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/utils.py
    https://en.wikipedia.org/wiki/Shoelace_formula

    """

    assert len(x) == len(y)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/utils.py
    """

    assert len(x) == len(y)
    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def convert_ls_polygonlabels_to_coco(
        annotation_id, image_id,
        points, width, height):
    """
    From https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/converter.py#L836
    """
    points_abs = [
        (x / 100 * width, y / 100 * height) for x, y in points
    ]
    x, y = zip(*points_abs)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,  # single category
        "segmentation":
            [
                [coord for point in points_abs for coord in point]
            ],
        "bbox": get_polygon_bounding_box(x, y),
        "ignore": 0,
        "iscrowd": 0,
        "area": get_polygon_area(x, y),
    }


def convert_coco_segmentation_to_ls(
    category_id, segmentation, categories, from_name, image_height, image_width, to_name
):
    """
    Modified from https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/imports/coco.py
     function name = create_segmentation
    Convert COCO segmentation annotation to Label Studio polygon format.

    COCO segmentation format: flat array of [x1,y1,x2,y2,...] coordinates
    Label Studio format: array of [x,y] points as percentages

    Args:
        category_id (int): COCO category ID for this segmentation
        segmentation (list): Flat list of polygon coordinates [x1,y1,x2,y2,...]
        categories (dict): Mapping of category_id to category name
        from_name (str): Control tag name from Label Studio labeling config
        image_height (int): Height of the source image in pixels
        image_width (int): Width of the source image in pixels
        to_name (str): Object name from Label Studio labeling config

    Returns:
        dict: Label Studio polygon annotation item
    """
    label = categories[int(category_id)]
    # Convert flat array [x1,y1,x2,y2,...] to array of points [[x1,y1],[x2,y2],...]
    points = [list(x) for x in zip(*[iter(segmentation)] * 2)]

    # Convert absolute coordinates to percentages
    for i in range(len(points)):
        points[i][0] = points[i][0] / image_width * 100.0
        points[i][1] = points[i][1] / image_height * 100.0

    item = {
        "id": uuid4().hex[0:10],
        "type": "polygonlabels",
        "value": {"points": points, "polygonlabels": [label]},
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
    }
    return item


def extract_bbox(a):
    return (a['x'], a['y'], a['width'], a['height'])


def filter_transform_record(row):
    """
    For object detections, make coco annotations and provide image path.
    """
    if not row['annotations']:
        return

    # replace with FILE_MOUNT for production
    # use file_mount for local dev
    # cwd = os.getcwd()
    # file_mount = cwd.replace('ultralytics', 'label-studio/mydata/stitchermedia')
    file_name = row['panorama_path'].replace('/media/', '')
    file_name = file_name.replace('/panorama', '_panorama')
    panorama_path = row['panorama_path'].replace('/media', '')
    row['panorama_path'] = FILE_MOUNT + panorama_path
    coco_annotations = [
        label_studio_to_coco(
            extract_bbox(a), a['original_width'], a['original_height']) for a in row['annotations']]
    row.update({
        'coco_annotations': coco_annotations,
        'file_name': file_name
    })
    return row


def filter_transform_segmentation_record(row, image_id, width, height, anno_size_gte, starting_anno_id):
    """
    For segmentation, make coco annotations and provide image path.
    """
    if not row['annotations_segment']:
        return
    anno_id = starting_anno_id + 1
    coco_annotations = [convert_ls_polygonlabels_to_coco(
        anno_id + i, image_id,
        v['points'], width, height) for i, v in enumerate(row['annotations_segment'])]
    if anno_size_gte:
        # filter out small annotations
        coco_annotations = [
            v for v in coco_annotations if v['bbox'][2] >= anno_size_gte or v['bbox'][3] >= anno_size_gte
        ]
    row.update({
        'coco_annotations': coco_annotations,
    })
    return row


def get_image_info(image_path, image_id):
    # PIL.Image.open is "lazy" - it reads metadata without loading all pixels
    with Image.open(image_path) as img:
        width, height = img.size
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height
    }


def yolo_to_coco_poly(yolo_poly, w, h):
    """Converts normalized YOLO [x, y, x, y...] to pixel-space [x, y, x, y...]"""
    return [coord * w if i % 2 == 0 else coord * h for i, coord in enumerate(yolo_poly)]


def calculate_polygon_area(xs, ys):
    """Calculates the area of a polygon using the Shoelace formula."""
    n = len(xs)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def get_yolo_parts_to_coco(parts, image_width, image_height):
    class_id = int(parts[0]) + 1
    poly_normalized = list(map(float, parts[1:]))

    # Convert to pixel coordinates
    poly_pixels = yolo_to_coco_poly(poly_normalized, image_width, image_height)

    # Calculate simple Bbox from polygon (min/max x, min/max y)
    xs = poly_pixels[0::2]
    ys = poly_pixels[1::2]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    width, height = x_max - x_min, y_max - y_min
    x_min = int(x_min)
    y_min = int(y_min)
    width = int(width)
    height = int(height)
    poly_area = round(calculate_polygon_area(xs, ys), 2)
    return {
        'width': width,
        'height': height,
        'poly_area': poly_area,
        'class_id': class_id,
        'poly_pixels': poly_pixels,
        'bbox': [x_min, y_min, width, height]
    }


def convert_yolo_to_coco(
        yolo_file,
        image_width,
        image_height,
        image_id,
        starting_anno_id=0,
        anno_size_gte=50):
    """
    Reads a YOLO segmentation .txt file
    and converts it to COCO format.
    """
    coco_results = []
    anno_id = starting_anno_id + 1
    with open(yolo_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or len(parts) < 7:  # A valid polygon needs at least 3 points (1 class + 6 coords)
                continue

            coco_parts = get_yolo_parts_to_coco(parts, image_width, image_height)

            # Filter out annotations smaller than the threshold or tiny area
            if anno_size_gte and (coco_parts['width'] < anno_size_gte or coco_parts['height'] < anno_size_gte):
                continue

            if coco_parts['poly_area'] < 1.0:
                continue

            coco_rec = {
                "id": anno_id,
                "image_id": image_id,
                "category_id": coco_parts['class_id'],
                "segmentation": [coco_parts['poly_pixels']],
                "area": coco_parts['poly_area'],
                "bbox": coco_parts['bbox'],
                "iscrowd": 0,
                "ignore": 0
            }

            # additional check, calculate area exactly as SAHI to filter additional unusuals.
            sahi_annot = CocoAnnotation.from_coco_annotation_dict(coco_rec)
            if sahi_annot.area < 1.0:
                print(f"WARNING: image_id {image_id} annotation id {anno_id} had an area of {sahi_annot.area}")
                continue

            coco_results.append(coco_rec)
            anno_id += 1

    return coco_results


def resize_imgs_in_dir(source_dir: str, output_folder_name: str, target_max_dim: int):
    print(f'resize_imgs_in_dir from {source_dir} to {output_folder_name}')
    source_path = Path(source_dir).resolve()
    output_path = source_path / output_folder_name
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory '{source_path}' does not exist.")
    # Allowed extensions (case-insensitive)
    valid_extensions = {".jpg", ".jpeg", ".png"}
    processed_count = 0
    # Iterate through files in the source directory
    for file_path in source_path.iterdir():
        # Skip directories and non-matching file extensions (ignoring the output folder itself)
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                print(f'copy and resize {file_path}')
                with Image.open(file_path) as img:
                    orig_w, orig_h = img.size
                    # Calculate target dimensions maintaining aspect ratio
                    if orig_w >= orig_h:
                        new_w = target_max_dim
                        new_h = max(1, int(orig_h * (target_max_dim / orig_w)))
                    else:
                        new_h = target_max_dim
                        new_w = max(1, int(orig_w * (target_max_dim / orig_h)))
                    # High-quality resize algorithm
                    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Handle image mode compatibility (e.g., RGBA PNG saved to JPG)
                    ext = file_path.suffix.lower()
                    if ext in {".jpg", ".jpeg"} and resized_img.mode in ("RGBA", "P"):
                        resized_img = resized_img.convert("RGB")
                    # Save resized image to the new output folder
                    save_destination = output_path / file_path.name
                    resized_img.save(save_destination)
                    processed_count += 1
            except Exception as e:
                print(f"✖ Failed to resize {file_path.name}: {e}")
    print(f"\nDone! Successfully resized {processed_count} image(s).")
    print(f"Saved to: {output_path}")


def filter_small_yolo_annotations(image_dir: str, label_dir: str, min_pixel_size: int):
    """
    Scans a directory for images and matching YOLO segmentation .txt files.
    Calculates the bounding box width and height in pixels for each annotation,
    and removes any annotation where either dimension is below min_pixel_size.
    """
    source_path = Path(image_dir).resolve()
    label_path = Path(label_dir).resolve()
    valid_extensions = {".jpg", ".jpeg", ".png"}
    if not source_path.exists():
        raise FileNotFoundError(f"Directory '{source_path}' does not exist.")
    if not label_path.exists():
        raise FileNotFoundError(f"Directory '{label_path}' does not exist.")

    processed_files = 0
    total_removed = 0
    # Iterate through images
    print(f"Filtering annotations smaller than {min_pixel_size}px.")
    for img_path in source_path.iterdir():
        if not (img_path.is_file() and img_path.suffix.lower() in valid_extensions):
            continue

        txt_path = label_path / img_path.with_suffix(".txt").name
        if not txt_path.exists():
            print(f'Warning: {txt_path} does not exist, skipping')
            continue

        # Get image dimensions to convert normalized coordinates to pixels
        with Image.open(img_path) as img:
            image_width, image_height = img.size
        # Read annotation lines
        with open(txt_path, "r") as f:
            lines = f.readlines()
        kept_lines = []
        removed_in_file = 0
        for line in lines:
            parts = line.strip().split()
            # YOLO segmentation line: class_id x1 y1 x2 y2 ... (at least 3 x/y pairs)
            if not parts or len(parts) < 7:
                continue

            coco_parts = get_yolo_parts_to_coco(parts, image_width, image_height)
            # Filter out annotations smaller than the threshold or tiny area
            if coco_parts['width'] < min_pixel_size or coco_parts['height'] < min_pixel_size:
                removed_in_file += 1
            else:
                kept_lines.append(line.strip() + "\n")

        # Overwrite the .txt file with the filtered annotations
        with open(txt_path, "w") as f:
            f.writelines(kept_lines)

        processed_files += 1
        total_removed += removed_in_file

    print(f"Total annotations removed below {min_pixel_size}px threshold: {total_removed}")


def remove_without_annotations(image_dir: str, label_dir: str):
    image_path = Path(image_dir).resolve()
    label_path = Path(label_dir).resolve()
    if not label_path.exists() or not image_path.exists():
        raise FileNotFoundError(
            "Provided image or label directory does not exist."
        )

    # Allowed image extensions to check for matching stems
    valid_img_extensions = {".jpg", ".jpeg", ".png"}
    total_files = 0
    removed_count = 0
    for file_path in label_path.glob("*.txt"):
        if file_path.is_file():
            total_files += 1
            content = file_path.read_text(encoding="utf-8")
            # .strip() removes whitespace, ensuring files with just spaces/newlines count as empty
            if not content.strip():
                # 1. Delete the empty label file
                file_path.unlink()

                # 2. Search for and delete the matching image file in image_dir
                image_found = False
                for ext in valid_img_extensions:
                    img_file = image_path / f"{file_path.stem}{ext}"
                    if img_file.exists():
                        img_file.unlink()
                        image_found = True
                        print(
                            f"🗑 Removed empty label '{file_path.name}' and image '{img_file.name}'"
                        )
                        break

                if not image_found:
                    print(
                        f"🗑 Removed empty label '{file_path.name}' (no matching image found in image_dir)"
                    )

                removed_count += 1

    print(f"Scanned {total_files} label file(s)")
    print(f"Removed {removed_count} unannotated pair(s).")
    return total_files - removed_count
