from os import listdir
from os.path import isfile, join, basename

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

# run inference example
IMG_DIR = 'my_directory'
img_files = [
    IMG_DIR + '/' + f for f in listdir(IMG_DIR) if isfile(join(IMG_DIR, f))
]

# Predict with the model
results = model(img_files)  # predict on images
for result in results:
    # change this to save to a diretory instead of at top level
    result.save(filename='inference_{0}'.format(basename(result.path)))
