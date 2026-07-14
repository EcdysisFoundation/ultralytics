from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('yolo26x-seg.pt')
    model.train(
        data='datasets/data.yaml',
        epochs=300,
        imgsz=640,
        patience=50,
        batch=8,
        workers=8,
        device=[0, 1]
    )
