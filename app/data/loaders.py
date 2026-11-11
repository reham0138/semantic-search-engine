
import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    """Load application records from a JSON file containing a list of objects."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Dataset must be a JSON array of objects.")

    return data
