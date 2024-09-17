from os import listdir
from os.path import isfile, join, basename

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  #'yolov8n.yaml' to use without pretrained weights or 'yolov8n.pt' for a pretrained model
results = model.train(data='dataset.yaml', epochs=100, imgsz=640)
