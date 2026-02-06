import json
import os
import torch

from .dataset import get_stitcher_data
from .sahi_segmentation import predict
from .utils import put_predictions


STITCHER_URL = 'http://ecdysis01.local:8090/'


def main():
    """
    SAHI inference
    """
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(torch.cuda.get_device_name(0))
    all_data = get_stitcher_data(STITCHER_URL)

    file_mount = '/pool1/srv/label-studio/mydata/stitchermedia'
    api_post_url = STITCHER_URL + 'update-predictions-coco/'
    anno_size_gte = 50  # limits minimum annotation bbox size

    dont_overwrite = False
    send_these = []  # example [str(i) for i in range(4111, 4131)]

    for d in all_data:
        # we use a name convention in first for characters, filter those
        if d['upload_dir_name'][:4] not in send_these:
            continue
        if d['panorama_path']:
            if dont_overwrite and d['predictions_coco']:
                continue
            p = file_mount + d['panorama_path']
            p = p.replace('/media', '')
            if os.path.exists(p):
                print(f'performing inference on {p}')
                coco_result, original_width, original_height = predict(p)
                # filter missing bbox
                coco_result = [v for v in coco_result if v['bbox']]
                # filter based on bbox size
                coco_result = [
                    v for v in coco_result if v['bbox'][2] >= anno_size_gte or v['bbox'][3] >= anno_size_gte
                ]
                prediction_result = json.dumps([{
                    'predictions': coco_result,
                    'original_width': original_width,
                    'original_height': original_height
                }])
                if coco_result:
                    put_predictions(
                        api_post_url,
                        d['guid'],
                        prediction_result)
            else:
                print('path not found')
                print(p)


if __name__ == '__main__':
    main()
