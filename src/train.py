from src.synthesis import containers, io, trainer
from core.config_schema import AppConfig
import pandas as pd


def run_training(config_path: str = r"config\config.yaml"):
    cfg = AppConfig.load(config_path)
    raw_h = pd.read_csv(cfg.train.raw_h_df, low_memory=False)
    raw_p = pd.read_csv(cfg.train.raw_p_df, low_memory=False)
    train_bundle = containers.TrainingBundle.construct_bundle(
        raw_h,
        raw_p,
        cfg.train.h_nodes,
        cfg.train.p_nodes,
        cfg.train.age_col,
        cfg.train.gender_col,
        cfg.train.h_id_col,
    )

    model_train = trainer.Trainer(config=train_bundle.config)

    h_model = model_train.train_model(
        train_data=train_bundle.household,
        type="H",
        dag_path=cfg.paths.household_dag,
        score=cfg.train.score,
    )

    p_model = model_train.train_model(
        train_data=train_bundle.person,
        type="P",
        dag_path=cfg.paths.person_dag,
        score=cfg.train.score,
    )

    io.save_bn_bundle(
        h_model=h_model,
        p_model=p_model,
        config=train_bundle.config,
        path=cfg.paths.bundle_file,
    )
