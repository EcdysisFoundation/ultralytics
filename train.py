from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('yolo11l.pt')
    model.train(
        data='datasets/data.yaml',
        epochs=300,
        imgsz=640,
        device=[0, 1]
    )
