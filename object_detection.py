from os import listdir
from os.path import isfile, join, basename

from ultralytics import YOLO
from sahi.predict import predict


trained_model_path = 'runs/detect/train3/weights/best.pt'
pretrained_model_path = 'runs/detect/train9/weights/best.pt'
trained_model = YOLO(trained_model_path)
pretrained_model = YOLO(pretrained_model_path)

LOCAL_FILES_FOLDER = 'secondimage'

# run inference example
IMG_DIR = 'local_files/' + LOCAL_FILES_FOLDER
img_files = [
    IMG_DIR + '/' + f for f in listdir(IMG_DIR) if isfile(join(IMG_DIR, f))
]
img_files = [i for i in img_files if basename(i)]

project='local_files/output/' + LOCAL_FILES_FOLDER
confidence = 0.1

# Standard predicton with the models
for img in img_files:
    trained_model.predict(source=img, save=True, conf=confidence, project=project, name='trained', exist_ok=True)
    pretrained_model.predict(source=img, save=True, conf=confidence, project=project, name='pretrained', exist_ok=True)

# SAHI prediction with the models
# These save to runs/predict/exp...

confidence = 0.3
heightwidth = 640
predict(
    model_type="ultralytics",
    model_path=trained_model_path,
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=confidence,
    source=IMG_DIR,
    slice_height=heightwidth,
    slice_width=heightwidth,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    visual_hide_labels=True,
    visual_hide_conf=True
)

predict(
    model_type="ultralytics",
    model_path=pretrained_model_path,
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=confidence,
    source=IMG_DIR,
    slice_height=heightwidth,
    slice_width=heightwidth,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    visual_hide_labels=True,
    visual_hide_conf=True
)
