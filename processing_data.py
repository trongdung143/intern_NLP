from pathlib import Path
import json
from tqdm import tqdm


def prcesssing_data():
    data_all = []

    for json_file in tqdm(list(Path("./data/json").glob("*.json")), desc="Read JSON"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
            data_all.append(data)

    TRAIN_DATA = []

    for item in data_all:
        TRAIN_DATA.append((item["annotations"][0][0], item["annotations"][0][1]))
    return TRAIN_DATA
