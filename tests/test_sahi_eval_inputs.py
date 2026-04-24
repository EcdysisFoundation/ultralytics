import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET_DIR = '/home/ecdysis/ultralytics/local_files/evaluation_dataset_1/'
DATASET_JSON = f'{DATASET_DIR}dataset_test.json'
EVAL_OUTPUT = f'{DATASET_DIR}evaluation_result.json'


def are_ids_matching():
    print('----- ARE_IDS_MATCHING -------')
    # Paths to your files
    gt_path = DATASET_JSON
    pred_path = EVAL_OUTPUT

    with open(gt_path, 'r') as f:
        gt = json.load(f)
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    # 1. Check Image IDs
    gt_ids = {img['id'] for img in gt['images']}
    pred_ids = {p['image_id'] for p in preds}
    matching_ids = gt_ids.intersection(pred_ids)

    print(f"GT Image IDs (sample): {list(gt_ids)[:5]}")
    print(f"Pred Image IDs (sample): {list(pred_ids)[:5]}")
    print(f"Common IDs found: {len(matching_ids)}")

    # 2. Check Category IDs
    gt_cats = {c['id'] for c in gt['categories']}
    pred_cats = {p['category_id'] for p in preds}
    matching_cats = gt_cats.intersection(pred_cats)

    print(f"GT Category IDs: {gt_cats}")
    print(f"Pred Category IDs: {pred_cats}")
    print(f"matching_cats: {matching_cats}")


def inspect_prediction():
    print('----------- INSPECT_PREDICTION ------------')
    pred_path = EVAL_OUTPUT

    with open(pred_path, 'r') as f:
        preds = json.load(f)

        if len(preds) > 0:
            first_pred = preds[0]
            print("--- Sample Prediction ---")
            for key, value in first_pred.items():
                # Truncate long segmentation lists for readability
                if key == 'segmentation' and isinstance(value, list):
                    print(f"{key}: List of length {len(value[0]) if value else 0}")
                else:
                    print(f"{key}: {value}")
        else:
            print("The prediction file is literally empty ([])!")


def check_ground_truth():
    print('-------- CHECK_GROUND_TRUTH ----------')
    with open(DATASET_JSON, 'r') as f:
        gt = json.load(f)

    # Look at the first annotation
    if gt['annotations']:
        ann = gt['annotations'][0]
        print(f"GT Has Segmentation: {'segmentation' in ann and len(ann['segmentation']) > 0}")
        if 'segmentation' in ann:
            print(f"GT Segmentation Type: {type(ann['segmentation'])}")
            print(f"GT Segmentation Nesting: {type(ann['segmentation'][0])}")


def check_boundaries():
    print('---------- CHECK_BOUNDARIES -----------')
    gt_path = DATASET_JSON
    pred_path = EVAL_OUTPUT

    with open(gt_path, 'r') as f:
        gt = json.load(f)
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    # Map image IDs to their dimensions
    img_dims = {img['id']: (img['width'], img['height']) for img in gt['images']}

    out_of_bounds_count = 0
    empty_bbox_count = 0
    for p in preds:
        img_id = p['image_id']
        if img_id in img_dims:
            if p['bbox']:
                w_limit, h_limit = img_dims[img_id]
                x, y, w, h = p['bbox']

                if (x + w > w_limit) or (y + h > h_limit) or (x < 0) or (y < 0):
                    out_of_bounds_count += 1
            else:
                empty_bbox_count += 1

    print(f"Total Predictions: {len(preds)}")
    print(f"Out of Bounds Predictions: {out_of_bounds_count}")
    print(f"Empty bbox predictions: {empty_bbox_count}")
    if len(gt['images']) > 0:
        print(f"Image 0 dimensions in GT: {img_dims.get(0)}")


def manual_load_test():
    gt_path = DATASET_JSON
    res_path = EVAL_OUTPUT

    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(res_path) # This is where it usually fails

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")


if __name__ == "__main__":
    are_ids_matching()
    inspect_prediction()
    check_ground_truth()
    check_boundaries()
    # if above checks out
    manual_load_test()
