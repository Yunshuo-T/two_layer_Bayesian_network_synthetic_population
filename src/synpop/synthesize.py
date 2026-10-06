from typing import Literal

from synpop import io
from synpop.calibration.matrix import CalibrationInput
from synpop.config import AppConfig
from synpop.calibration.matrix import CalibrationMatrixBuilder as matrix_builder
from synpop.calibration.calibrator import CALIBRATOR
from synpop.synthesis import generator


def run_synthesize(
    calibration: Literal["ipu", "gr", "hipf", "cross_entropy", None],
    config_path: str = r"config\config.yaml",
    joint_attributes: list[str] | None = None,
    # save_results: bool = True
):

    cfg = AppConfig.load(config_path)

    bundle = io.load_bn_bundle(cfg.paths.bundle_file)
    h_model = bundle["h_model"]
    p_model = bundle["p_model"]
    bn_config = bundle["config"]

    two_layer_bn = generator.Generator(
        h_model=h_model,
        p_model=p_model,
        config=bn_config,
        p_data_include_kid=cfg.synthesis.p_data_include_kid,
    )

    h_bn, p_bn = two_layer_bn.synthesize(
        num_households=cfg.synthesis.num_households,
        max_workers=cfg.synthesis.max_workers,
        random_seed=cfg.synthesis.random_seed,
        pool_multiplier=cfg.synthesis.pool_multiplier,
        max_retries=cfg.synthesis.max_retries,
    )

    io.save_csv(h_bn, cfg.paths.output_dir, "h_bn")
    io.save_csv(p_bn, cfg.paths.output_dir, "p_bn")

    if calibration:
        calibration_input = CalibrationInput(
            h_df=h_bn,
            p_df=p_bn,
            h_marginals=cfg.calibration.h_marginals,
            p_marginals=cfg.calibration.p_marginals,
            h_attrs=cfg.calibration.h_attributes,
            p_attrs=cfg.calibration.p_attributes,
            h_id_col=cfg.train.h_id_col,
        )

        calibration_matrix = matrix_builder.build(
            calibration_input, joint_attributes=joint_attributes
        )

        calibrator = CALIBRATOR[calibration]
        calibration_result = calibrator(calibration_matrix).fit(
            max_iter=cfg.calibration.max_iterations,
            tolerance=cfg.calibration.tolerance,
            **cfg.calibration.kwargs,
        )
        h_id = cfg.train.h_id_col
        h_bn["weight"] = (
            calibration_result.weights
        )  # FIXME  household rows can have a different order, assigning weights to the wrong households.can be fixed by storing IDs in CalibrationMatrix
        p_bn["weight"] = p_bn[h_id].map(h_bn.set_index(h_id)["weight"])
        io.save_csv(h_bn, cfg.paths.output_dir, "h_calibrated")
        io.save_csv(p_bn, cfg.paths.output_dir, "p_bn_calibrated")
        return calibration_result
