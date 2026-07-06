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


def eval_test_dataset(eval_dirs):
    """
        eval_dirs = {
            'images_path': '/eval_dataset_pano/images/test',
            'labels_path': '/eval_dataset_pano/labels/test'
        }
    """

    stats = get_label_stats(eval_dirs['labels_path'])
    print(f"Class distribution in labels_path: {stats}")


if __name__ == '__main__':
    eval_test_dataset()
