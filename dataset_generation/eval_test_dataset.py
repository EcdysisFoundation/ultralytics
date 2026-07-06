import os
import yaml
from collections import Counter
from pathlib import Path

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


def eval_test_dataset(eval_dirs):
    stats = get_label_stats(eval_dirs['labels_path'])
    print(f"Class distribution in labels_path: {stats}")

    create_yaml(eval_dirs['dataset_dir'])


if __name__ == '__main__':
    # to test ..
    curr_dir = os.getcwd()
    eval_dirs = {
            'dataset_dir': 'eval_dataset_pano',
            'images_path': f'{curr_dir}/eval_dataset_pano/images/test',
            'labels_path': f'{curr_dir}/eval_dataset_pano/labels/test'
        }
    eval_test_dataset(eval_dirs)
