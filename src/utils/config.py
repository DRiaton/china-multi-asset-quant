
from pathlib import Path

import pandas as pd
import yaml


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dates = config["dates"]
    warmup, start, end = map(pd.Timestamp, (dates["warmup_start"], dates["start"], dates["end"]))
    if not warmup < start < end:
        raise ValueError("Expected warmup_start < start < end")
    return config
