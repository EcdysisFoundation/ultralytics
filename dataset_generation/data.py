import json
import pandas as pd
from pathlib import Path


class ObjectDetectData:

    img_mount = '/pool1/srv/bugbox3/bugbox3/media/'
    obj_det_export = './local_files/obj_det_selections.json'

    def get_full_df(self):
        d = pd.read_json(self.obj_det_export)
        p = Path(self.img_mount)
        d['full_image_path'] = d['image_thumbnail_large'].apply(lambda x: str(p / x))
        return d
