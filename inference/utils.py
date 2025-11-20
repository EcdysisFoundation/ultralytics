import requests


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
