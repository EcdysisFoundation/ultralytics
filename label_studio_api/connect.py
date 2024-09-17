from label_studio_sdk.client import LabelStudio

# Define the URL where Label Studio is accessible and the API key for your user account


LABEL_STUDIO_URL = 'http://http://10.147.19.124/:8090'

# API_KEY for app on a private network. Change and do not commit to deployed to public net
# API key is available at the Account & Settings > Access Tokens page in Label Studio UI
API_KEY = 'a5aa0b32b4ebab17be5df1e948dec2da716000b3'


# Connect to the Label Studio API and check the connection
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Import ls from or in some other file to pull in an export or some other task
# see https://labelstud.io/api
