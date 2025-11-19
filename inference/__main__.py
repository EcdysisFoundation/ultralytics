import os

from .dataset import get_stitcher_data
from .sahi_segmentation import predict


stitcher_url = 'http://ecdysis01.local:8090/'


def main():
    """
    SAHI inference
    """

    # all_data = get_stitcher_data(stitcher_url)
    add_data = [
        {'panorama_path': '/media/9d5e4dc2-d63d-4060-b9ce-b199aced7243/panorama.jpg'}]
    file_mount = '/pool1/srv/label-studio/mydata/stitchermedia'
    dont_overwrite = True


    for d in all_data:
        #if d['guid'] not in send_these:
        #    continue
        if d['panorama_path']:
            #if dont_overwrite and d['predictions']:
            #    continue
            p = file_mount + d['panorama_path']
            p = p.replace('/media', '')
            if os.path.exists(p):
                print(f'performing inference on {p}')
                prediction = predict(p)
                #if prediction:
                #    put_predictions(stitcher_url, d['guid'], prediction)
            else:
                print('path not found')
                print(p)


if __name__ == '__main__':
    main()
