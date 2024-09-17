import shutil
from os import listdir
from os.path import isfile, join

# edit to current export working with. Use Yolo format export
EXPORT_DIR = 'label_studio_exports/project-4-at-2024-09-03-16-43-1214bb82'

# change this to dynamically pick 20% of images per class
# instead here we are setting at 20 because we expect each class has 100 images
MAX_VAL_COUNT = 20

classes = []

# class_counts = {
#   'label': {'val_count': 0, 'train_count': 0},
#   'label_2': {'val_count': 0, 'train_count': 0}, ...
# }
class_counts = {}

# read classes
with open(EXPORT_DIR + '/classes.txt') as c:
    lines = c.readlines()
    for line in lines:
        classes.append(line.replace('\n', ''))
print('classes read as ...')
print(classes)


LABEL_DIR = EXPORT_DIR + '/labels'
IMAGE_DIR = EXPORT_DIR + '/images'
# may need to require it be a .txt if other files in the directory
label_files = [
    f for f in listdir(LABEL_DIR) if isfile(join(LABEL_DIR, f))
]

print('Structuring dataset for {0} files'.format(len(label_files)))
for file in label_files:
    label_filepath = LABEL_DIR + '/' + file
    image_file = file.replace('.txt', '.JPG')
    image_filepath = IMAGE_DIR + '/' + image_file
    with open(label_filepath) as f:
        for line in f.readlines():
            if len(line):
                theclass = line.split(' ')[0]
                if theclass not in class_counts.keys():
                    class_counts[theclass] = {'val_count': 0, 'train_count': 0}
                if class_counts[theclass]['val_count'] < MAX_VAL_COUNT:
                    class_counts[theclass]['val_count'] += 1
                    shutil.copyfile(label_filepath, 'datasets/val/labels/' + file)
                    shutil.copyfile(
                        image_filepath, 'datasets/val/images/' +
                        image_file)
                else:
                    class_counts[theclass]['train_count'] += 1
                    shutil.copyfile(label_filepath, 'datasets/train/labels/' + file)
                    shutil.copyfile(
                        image_filepath, 'datasets/train/images/' +
                        image_file)

for v, items in class_counts.items():
    print(v)
    print(items)
