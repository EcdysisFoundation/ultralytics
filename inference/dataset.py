import requests


def get_stitcher_data(stitcher_url):
    api_list_url = stitcher_url + 'list-upload-files/'
    all_data = []
    offset = 0
    limit = 100

    while True:

        params = {
            'offset': offset,
            'limit': limit,
            'label_studio_filter': True}

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
