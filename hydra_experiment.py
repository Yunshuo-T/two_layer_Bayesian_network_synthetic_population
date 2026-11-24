import hydra
from omegaconf import DictConfig

from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone
from uuid import uuid4

from hydra.utils import to_absolute_path
from synpop.io import load_bn_bundle
from synpop.synthesis.generator import Generator
import logging
import sqlite3
import pandas as pd



def store_db(h_bn:pd.DataFrame,p_bn:pd.DataFrame,cfg: DictConfig):
    run_id = uuid4().hex

    # Serialize both DataFrames before opening the database connection.
    household_blob = h_bn.to_parquet(
        index=False,
        engine="pyarrow",
    )
    individual_blob = p_bn.to_parquet(
        index=False,
        engine="pyarrow",
    )

    db_path = Path(to_absolute_path("output/experiments.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS Bayesian_runs (
                    run_id TEXT PRIMARY KEY,
                    time TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    seed INTEGER,
                    household BLOB NOT NULL,
                    individual BLOB NOT NULL
                )
            """)

            conn.execute(
                """
                INSERT INTO Bayesian_runs (
                    run_id,
                    time,
                    threshold,
                    seed,
                    household,
                    individual
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    float(cfg.synthesis.threshold),
                    cfg.synthesis.random_seed,
                    household_blob,
                    individual_blob,
                ),
            )


@hydra.main(config_path="config",config_name="config",version_base=None)
def experiments(cfg:DictConfig):
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    
    bundle = load_bn_bundle(cfg.paths.bundle_file)
    h_model = bundle["h_model"]
    p_model = bundle["p_model"]
    bn_config = bundle["config"]

    two_layer_bn = Generator(
        h_model=h_model,
        p_model=p_model,
        config=bn_config,
        p_data_include_kid=cfg.synthesis.p_data_include_kid,
    )
    
    h_bn, p_bn = two_layer_bn.synthesize(
        total_population=cfg.synthesis.total_population,
        num_households=cfg.synthesis.num_households,
        max_workers=cfg.synthesis.max_workers,
        random_seed=cfg.synthesis.random_seed,
        pool_multiplier=cfg.synthesis.pool_multiplier,
        max_retries=cfg.synthesis.max_retries,
        threshold=cfg.synthesis.threshold,
    )

    store_db(h_bn=h_bn,p_bn=p_bn,cfg=cfg)


if __name__ == "__main__":
    experiments()