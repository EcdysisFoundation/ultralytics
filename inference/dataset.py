import os
import shutil
import requests


def get_stitcher_data(stitcher_url):
    api_list_url = stitcher_url + 'list-upload-files/'
    all_data = []
    offset =  0
    limit = 100


    while True:

        params = {'offset': offset, 'limit': limit}

        try:
            response = requests.get(api_list_url, params=params)
        except Exception as e:
            print(e)
            break

        if response.status_code == 200:
            data = response.json()
            if not data:
                break

            all_data.extend(data)

            offset += limit
        else:
            print(f"Error: {response.status_code}")
            break

    print(f"Retrieved {len(all_data)} items.")

    return all_data


# remove try direct FUSE link
def create_dataset_directory(all_data, stitcher_dir):

    img_mount = '/pool1/srv/stitcher/media/'

    # clear previous directory
    if os.path.exists(stitcher_dir):
        shutil.rmtree(stitcher_dir)  # Remove the existing directory and its contents
        print(f"Removed existing directory: {stitcher_dir}")

    os.makedirs(stitcher_dir)  # Create the new directory
    print(f"Created new directory: {stitcher_dir}")

    # populate with symlinks
    for d in all_data:
        p = stitcher_dir + os.path.basename(d['panorama_path'])
    print('done create_dataset_directory')
