from pathlib import Path
import json
from tqdm import tqdm


def check_name_no_email():
    files = sorted(Path("./data/json").glob("*.json"), key=lambda x: int(x.stem))

    for json_file in tqdm(files, desc="Check JSON"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        text = data["annotations"][0][0]
        ents = data["annotations"][0][1]["entities"]
        labels = [label for _, _, label in ents]

        has_name = "NAME" in labels
        has_email = "EMAIL" in labels

        if (
            has_email and not has_name
        ):  # or (has_name and not has_email and "@" in text):
            print(f"⚠ {json_file.name}")
            print(f"   Content: {text[:200]}...")
            print(f"   Entities: {ents}\n")


check_name_no_email()
