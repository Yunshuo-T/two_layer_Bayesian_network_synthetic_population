import pandas as pd

from synpop import io
from synpop.config import AppConfig
from synpop.training import trainer


def run_training(config_path: str = r"config\config.yaml"):
    cfg = AppConfig.load(config_path)
    if cfg.train.raw_h_df.endswith('.csv'):
        raw_h = pd.read_csv(cfg.train.raw_h_df, low_memory=False)
        raw_p = pd.read_csv(cfg.train.raw_p_df, low_memory=False)
    else:
        raw_h = pd.read_excel(cfg.train.raw_h_df)
        raw_p = pd.read_excel(cfg.train.raw_p_df)
    train_bundle = trainer.TrainingBundle.construct_bundle(
        raw_h_df=raw_h,
        raw_p_df=raw_p,
        h_nodes=cfg.train.h_nodes,
        p_nodes=cfg.train.p_nodes,
        age_col=cfg.train.age_col,
        gender_col=cfg.train.gender_col,
        h_id_col=cfg.train.h_id_col,
        composition_cols=cfg.train.composition_cols,
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
