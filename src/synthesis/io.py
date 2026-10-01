import pickle
from src.synthesis.containers import ModelSchema
from pathlib import Path
import pandas as pd


def save_bn_bundle(
    h_model,
    p_model,
    config: ModelSchema,
    path: str,
) -> None:
    bundle = {
        "h_model": h_model,
        "p_model": p_model,
        "config": config,
    }

    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bn_bundle(path: str):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def save_csv(df: pd.DataFrame, directory: str | Path, filename: str, **kwargs):
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    filename = filename.rstrip(".csv")
    file_path = dir_path / f"{filename}.csv"

    counter = 1
    while file_path.exists():
        file_path = dir_path / f"{filename}_{counter}.csv"
        counter += 1
    df.to_csv(file_path, **kwargs)
