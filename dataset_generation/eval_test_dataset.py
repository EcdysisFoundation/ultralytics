import json
import os
import yaml
from collections import Counter
from pathlib import Path
from PIL import Image

CLASS_NAMES = {
    0: "Arthropod",
}  # as they appear in YOLO
# shift by one. YOLO starts at 0 but some COCO formats ignore 0
CLASS_NAMES_COCO = {i + 1: name for i, name in CLASS_NAMES.items()}


def get_label_stats(label_dir):
    stats = Counter()
    for label_file in Path(label_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = line.split()[0]
                class_name = CLASS_NAMES_COCO[int(class_id) + 1]
                stats[class_name] += 1
    return stats


def create_yaml(dataset_dir):
    """
    Generates the data.yaml file required for YOLO training.
    """
    dataset_root = Path(dataset_dir)
    yaml_content = {
        'path': str(dataset_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': CLASS_NAMES_COCO
    }

    yaml_path = dataset_root / 'data.yaml'

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully created metadata: {yaml_path}")


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


def convert_to_coco_lite(images_dir, labels_dir, output_json, class_names):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in class_names.items()]
    }

    ann_id = 1
    image_files = sorted(list(Path(images_dir).glob("*.jpg")))

    for i, img_path in enumerate(image_files):
        print(f"Processing {img_path.name}...")

        img_info = get_image_info(img_path, i)
        coco["images"].append(img_info)

        label_path = Path(labels_dir) / img_path.with_suffix('.txt').name
        if not label_path.exists():
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0]) + 1
                poly_normalized = parts[1:]

                poly_pixels = yolo_to_coco_poly(poly_normalized, img_info["width"], img_info["height"])

                # Calculate simple Bbox from polygon (min/max x, min/max y)
                xs = poly_pixels[0::2]
                ys = poly_pixels[1::2]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                width, height = x_max - x_min, y_max - y_min

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": class_id,
                    "segmentation": [poly_pixels],
                    "area": width * height, # Simplified area
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"Done! Created {output_json}")


def eval_test_dataset(eval_dirs):
    stats = get_label_stats(eval_dirs['labels_path'])
    print(f"Class distribution in labels_path: {stats}")

    create_yaml(eval_dirs['dataset_dir'])

    convert_to_coco_lite(
        eval_dirs['images_path'],
        eval_dirs['labels_path'],
        f"{eval_dirs['dataset_dir']}/dataset_test.json",
        CLASS_NAMES_COCO)


if __name__ == '__main__':
    # to test ..

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    curr_dir = os.getcwd()
    eval_dirs = {
            'dataset_dir': 'eval_dataset_pano',
            'images_path': f'{curr_dir}/eval_dataset_pano/images/test',
            'labels_path': f'{curr_dir}/eval_dataset_pano/labels/test'
        }
    eval_test_dataset(eval_dirs)
