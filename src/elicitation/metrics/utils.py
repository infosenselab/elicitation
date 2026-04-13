import json
from pathlib import Path
import numpy as np


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def load_dialogues(base_path):
    """
    Yield dialogues from *all* JSON files under base_path (recursive).
    Handles both [dialogue, ...] and {"dialogues": [...]} formats.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"data path does not exist: {base_path}")
    if not base.is_dir():
        raise NotADirectoryError(f"data path is not a directory: {base_path}")
    
    json_files = list(Path(base_path).rglob("*.json"))
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Case 1: top-level list
            if isinstance(data, list):
                for d in data:
                    yield d

            # Case 2: dict with key 'dialogues'
            elif isinstance(data, dict) and "dialogues" in data and isinstance(data["dialogues"], list):
                for d in data["dialogues"]:
                    yield d

        except Exception as e:
            print(f"Error in {json_file}: {e}")
