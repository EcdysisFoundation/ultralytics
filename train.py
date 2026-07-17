from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('yolo26x-seg.pt')
    # see configuration settings https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    model.train(
        data='datasets/data.yaml',
        epochs=400,
        patience=75,
        imgsz=1024,
        batch=4,
        workers=4,
        device=[0, 1],
        single_cls=True,
        mask_ratio=1,  # use full resolution masks, dont down sample to reduce memory usage
        overlap_mask=False,  # dont merge overlapping objects
        cos_lr=True,  # learning rate decay to Cosine Annealing scheduler for maximum precision
        cls=0.3,  # Lower classification priority (since it's a single class)
        # cls offset adjustments to heavily prioritize spatial/mask accuracy
        box=10.0,  # or 12.0 Force highly accurate bounding boxes for mask cropping
        dfl=2.0,  # Sharpen fine-grained edge regression for fine details
        # end cls offset
        # augmentations
        degrees=180.0,  # allow any angle
        fliplr=0.5,  # horizontal flip probability
        flipud=0.5,  # vertical flip probability
        mosaic=0.0,  # turn off mosaic augmentation
        # try these agumentation for debris and glare
        # erasing=0.4,  # random small patch
        # scale=0.3,  # random zoom in and out
        # hsv_v=0.3,  # adjust brightness
        # hsv_s=0.3  # adjust saturation
    )
