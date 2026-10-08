from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf


@dataclass
class SynthesisConfig:
    num_households: int
    pool_multiplier: int
    max_retries: int
    random_seed: int
    max_workers: int
    p_data_include_kid: bool


@dataclass
class CalibrationConfig:
    max_iterations: int
    tolerance: float
    p_marginals: dict
    h_marginals: dict
    p_attributes: list
    h_attributes: list
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathConfig:
    output_dir: str
    dags_dir: str
    person_dag: str
    household_dag: str
    bundle_file: str


@dataclass
class TrainConfig:
    raw_h_df: str
    raw_p_df: str
    h_nodes: list[str]
    p_nodes: list[str]
    age_col: str
    age_bin: list
    gender_col: str
    h_id_col: str
    score: str


@dataclass
class AppConfig:
    train: TrainConfig
    synthesis: SynthesisConfig
    calibration: CalibrationConfig
    paths: PathConfig

    @classmethod
    def load(cls, yaml_path: str = r"config\config.yaml") -> "AppConfig":
        schema = OmegaConf.structured(cls)
        yaml = OmegaConf.load(Path(yaml_path))
        merged = OmegaConf.merge(schema, yaml)
        cfg = OmegaConf.to_object(merged)

        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)  # type: ignore
        Path(cfg.paths.dags_dir).mkdir(parents=True, exist_ok=True)  # type: ignore

        return cfg  # type: ignore
