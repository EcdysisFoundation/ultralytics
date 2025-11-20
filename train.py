from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('runs/detect/train5/weights/best.pt')
    model.train(
        data='datasets/data.yaml',
        epochs=300,
        imgsz=640,
        device=[0, 1]
    )
