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

    dont_overwrite = True
    send_these = ['4056', '4062', '4063', '4065', '4067']  # site numbers

    for d in all_data:
        if d['upload_dir_name'][:4] not in send_these:
            continue
        if d['panorama_path']:
            if dont_overwrite and d['predictions_coco']:
                continue
            p = file_mount + d['panorama_path']
            p = p.replace('/media', '')
            if os.path.exists(p):
                print(f'performing inference on {p}')
                prediction = predict(p)
                if prediction:
                    put_predictions(api_post_url, d['guid'], prediction)
            else:
                print('path not found')
                print(p)


if __name__ == '__main__':
    main()
