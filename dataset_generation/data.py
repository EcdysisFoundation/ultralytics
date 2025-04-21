import json
import pandas as pd
from pathlib import Path


class ObjectDetectData:

    img_mount = '/pool1/srv/bugbox3/bugbox3/media/'
    obj_det_export = './local_files/obj_det_selections.json'

    def load_json(self):

        try:
            with open(self.obj_det_export, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"Error: File not found: {self.obj_det_export}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in: {self.obj_det_export}")
            return None

    def get_full_df(self):
        d = pd.json_normalize(self.load_json())
        p = Path(self.img_mount)
        d['full_image_path'] = d['image'].apply(lambda x: str(p / x))
        return d

    def get_image_df(self):
        d = self.get_full_df()
        fields = ['id', 'specimen_id', 'full_image_path', 'specimen__classification__gbif_order']
        return d[fields].drop_duplicates()

