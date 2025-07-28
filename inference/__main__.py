import os

from .dataset import get_stitcher_data
from .sahi_stitched import predict, put_predictions


stitcher_url = 'http://ecdysis01.local:8090/'


def main():

    all_data = get_stitcher_data(stitcher_url)
    file_mount = '/pool1/srv/stitcher'

    for d in all_data:
        if d['panorama_path']:
            p = file_mount + d['panorama_path']

            if os.path.exists(p):
                prediction = predict(p)
                print(prediction)

            else:
                print('path not found')
                print(p)


if __name__ == '__main__':
    main()
