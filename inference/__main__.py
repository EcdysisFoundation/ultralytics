import os

from .dataset import get_stitcher_data
from .sahi_stitched import predict, put_predictions


stitcher_url = 'http://ecdysis01.local:8090/'


def main():

    all_data = get_stitcher_data(stitcher_url)
    file_mount = '/pool1/srv/stitcher'
    counter = 0 # stop early while testing
    for d in all_data:
        if counter == 1:
            break
        if d['panorama_path']:
            p = file_mount + d['panorama_path']

            print(p)

            if os.path.exists(p):
                print('path exists')
            else:
                print('path not found')

            # Remove local paths, use url instead
            """
            if local_dev_testing:
                p = stitcher_dir + d['panorama_path']
            else:
                p = stitcher_dir + os.path.basename(d['panorama_path'])
            if os.path.exists(p):
                predictions = predict(p)
                put_predictions(d['guid'], predictions)
                counter += 1
            else:
                print('panorama file does not exist: {0}'.format(p))
            """
            #prediction = predict(img_url)
            #print(prediction)


if __name__ == '__main__':
    main()
