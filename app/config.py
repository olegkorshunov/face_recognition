from dataclasses import dataclass

import yaml

FILE_PATH = "./config.yaml"


@dataclass
class _CFG:
    predictor_path: str
    checkpoint_path: str
    emb_dim: int


CFG = _CFG(**yaml.load(open(FILE_PATH, "r", encoding="utf-8"), Loader=yaml.FullLoader))
