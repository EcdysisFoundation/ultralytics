import cv2
import requests
from math import sqrt


def put_predictions(api_post_url, guid, predictions):

    params = {'guid': str(guid)}

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            api_post_url,
            params=params,
            data=predictions,
            headers=headers)
        if response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        else:
            print('Response returned None')
    except Exception as e:
        print(e)


def save_labeling_img(path, full_img_save_path):
    """
    Resizes for labeling app, below Decompression Bomb threshold.
    Saves to labeling directory even if not resized.
    """

    try:
        img = cv2.imread(path)
    except cv2.error as e:
        print(f"Error loading image: {e}")
        img = None
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    max_pixels = 128_000_000  # Decompression Bomb warning threshold
    margin = 0.95
    target_pixels = margin * max_pixels

    h, w = img.shape[:2]
    pixels = w * h
    print(f'img pixels are {pixels}')
    if pixels > target_pixels:
        print(f'resizing {path}')
        scale = sqrt(target_pixels / pixels)
        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        success = cv2.imwrite(full_img_save_path, resized)
    else:
        success = cv2.imwrite(full_img_save_path, img)

    if not success:
        raise IOError(f"Could not write image: {full_img_save_path}")
    return
