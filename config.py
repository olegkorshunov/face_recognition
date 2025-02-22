from dataclasses import dataclass

import yaml

FILE_PATH = "./config.yaml"


@dataclass
class _CFG:
    predictor_path: str
    identity_path: str
    img_folder_src: str
    img_folder_dst: str
    upsample: int
    crop_size: int

    label_trashold:int 


CFG = _CFG(**yaml.load(open(FILE_PATH, "r", encoding="utf-8"), Loader=yaml.FullLoader))
