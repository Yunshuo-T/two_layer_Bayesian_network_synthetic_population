"""Unit tests for synpop. Run from the project root with ``python -m pytest -q``.

Fixtures use small, deterministic datasets and Bayesian networks. File tests
write only inside pytest's temporary directories.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from pandas.testing import assert_frame_equal, assert_series_equal
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from synpop.calibration.calibrator import CrossEntropy, GR, HIPF, IPU
from synpop.calibration.matrix import (
    CalibrationInput,
    CalibrationMatrixBuilder,
    CalibrationResult,
)
from synpop.config import AppConfig, CalibrationConfig, TrainConfig
from synpop.io import load_bn_bundle, save_bn_bundle, save_csv
from synpop.schema import ColumnNames, ModelSchema
from synpop.synthesis.candidate import CandidatePool
from synpop.synthesis.evidence import PersonSimulationKey
from synpop.synthesis.generator import Generator, _batch_sample_persons
from synpop.training.trainer import Trainer, TrainingBundle
from synpop.utils.encoding import cate_codes, make_code_crosswalk, map_dataframe
from synpop.utils.seeding import random_seeds
from synpop.validation import Validation, compute_metrics


@pytest.fixture
def raw_data():
    """Three households with adults, a minor and a young child, in mixed order."""
    households = pd.DataFrame(
        {"household_id": [30, 10, 20], "tenure": ["owned", "rented", "owned"]}
    )
    people = pd.DataFrame(
        {
            "household_id": [20, 10, 20, 20, 30, 30],
            "age": [12, 65, 40, 4, 25, 75],
            "gender": ["M", "F", "F", "M", "F", "M"],
            "occupation": ["student", "retired", "worker", "child", "worker", "retired"],
        }
    )
    return households, people


@pytest.fixture
def schema():
    """Head and person encodings intentionally differ to exercise crosswalks."""
    code_to_attr = {
        ColumnNames.HEAD_AGE: {0: 5},
        ColumnNames.AGE_CATE: dict(enumerate(range(9))),
        ColumnNames.HEAD_GENDER: {0: "F"},
        "gender": {0: "M", 1: "F"},
    }
    attr_to_code = {
        attr: {value: code for code, value in values.items()}
        for attr, values in code_to_attr.items()
    }
    return ModelSchema(code_to_attr, attr_to_code, "gender", "household_id")


def _independent_model(states):
    """A real fitted network with uniform, independent categorical variables."""
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(states)
    model.add_cpds(
        *[
            TabularCPD(
                variable=variable,
                variable_card=len(values),
                values=np.full((len(values), 1), 1 / len(values)),
                state_names={variable: values},
            )
            for variable, values in states.items()
        ]
    )
    assert model.check_model()
    return model


@pytest.fixture
def generator(schema):
    household_states = {
        ColumnNames.H_TYPE: [0],
        ColumnNames.HEAD_AGE: [0],
        ColumnNames.HEAD_GENDER: [0],
        ColumnNames.ADULTS: [1],
        ColumnNames.MINORS: [1],
        ColumnNames.KIDS: [1],
    }
    person_states = {
        ColumnNames.H_TYPE: [0],
        ColumnNames.HEAD_AGE: [0],
        ColumnNames.HEAD_GENDER: [0],
        ColumnNames.MEMBER_RANK: [0, 1, 2],
        ColumnNames.AGE_CATE: [0, 1, 5],
        "gender": [0, 1],
    }
    return Generator(
        _independent_model(household_states), _independent_model(person_states), schema
    )


@pytest.fixture
def blueprint():
    return pd.DataFrame(
        {
            "household_id": [10, 20],
            ColumnNames.H_TYPE: [0, 0],
            ColumnNames.HEAD_AGE: [0, 0],
            ColumnNames.HEAD_GENDER: [0, 0],
            ColumnNames.ADULTS: [1, 1],
            ColumnNames.MINORS: [1, 1],
            ColumnNames.KIDS: [1, 1],
        }
    )


@pytest.fixture
def calibration_input():
    # IDs are deliberately unsorted: design rows must align by household ID.
    return CalibrationInput(
        h_df=pd.DataFrame({"household_id": [20, 10], "tenure": ["rented", "owned"]}),
        p_df=pd.DataFrame(
            {"household_id": [20, 10, 10], "gender": ["M", "F", "M"]}
        ),
        h_marginals={"tenure": {"owned": 4, "rented": 2}},
        p_marginals={"gender": {"F": 4, "M": 6}},
        h_attrs=["tenure"],
        p_attrs=["gender"],
        h_id_col="household_id",
    )


class TestEncoding:
    def test_category_maps_are_inverses_without_changing_input(self):
        df = pd.DataFrame({"gender": ["M", "F", "M"], "age": [40, 20, 60]})
        original = df.copy(deep=True)
        code_to_attr, attr_to_code = cate_codes(df, ["gender"])

        assert code_to_attr == {"gender": {0: "F", 1: "M"}}
        assert attr_to_code == {"gender": {"F": 0, "M": 1}}
        assert_frame_equal(df, original)
        encoded = map_dataframe(attr_to_code, df)
        assert encoded["gender"].tolist() == [1, 0, 1]
        assert_frame_equal(map_dataframe(code_to_attr, encoded), original)

    def test_explicit_category_order_includes_unobserved_categories(self):
        df = pd.DataFrame({"level": ["high", "low"]})
        code_to_attr, attr_to_code = cate_codes(
            df, ["level"], {"level": ["low", "medium", "high"]}
        )
        assert code_to_attr["level"] == {0: "low", 1: "medium", 2: "high"}
        assert attr_to_code["level"]["high"] == 2

    def test_mapping_skips_absent_columns_and_marks_unknown_values(self):
        df = pd.DataFrame({"level": ["known", "unknown"], "age": [20, 30]})
        result = map_dataframe({"level": {"known": 3}, "absent": {"a": 1}}, df)
        assert result.loc[0, "level"] == 3
        assert pd.isna(result.loc[1, "level"])
        assert_series_equal(result["age"], df["age"])
        assert df["level"].tolist() == ["known", "unknown"]

    def test_crosswalk_uses_semantic_values(self, schema):
        assert make_code_crosswalk(
            ColumnNames.HEAD_GENDER, "gender", schema.code_to_attr, schema.attr_to_code
        ) == {0: 1}

    def test_crosswalk_rejects_missing_target_category(self):
        with pytest.raises(ValueError, match="does not exist"):
            make_code_crosswalk("head", "person", {"head": {0: "F"}}, {"person": {"M": 0}})






class TestEvidenceAndCandidates:
    @pytest.mark.parametrize(
        ("rank", "group", "age", "gender"),
        [(0, "adult", 5, 1), (1, "adult", None, None),
         (1, "minor", 1, None), (2, "kid", 0, None)],
    )
    def test_evidence_constraints(self, rank, group, age, gender):
        key = PersonSimulationKey.make_evidence_key(3, rank, 0, 0, group, {0: 5}, {0: 1})
        assert key == PersonSimulationKey(3, rank, 0, 0, age, gender)
        assert {key: 2}[key] == 2

    @pytest.mark.parametrize(
        ("group", "age", "previous_age", "expected"),
        [("adult", 2, None, True), ("adult", 1, None, False),
         ("adult", 5, 4, False), ("adult", 4, 4, True),
         ("minor", 1, 2, True), ("minor", 0, None, False),
         ("kid", 0, 1, True), ("kid", 1, None, False)],
    )
    def test_candidate_age_constraints(self, group, age, previous_age, expected):
        assert CandidatePool._matches(group, age, previous_age) is expected

    def test_candidates_are_taken_oldest_first_and_consumed(self):
        pool = CandidatePool({"key": [{"age": 2}, {"age": 6}, {"age": 4}]}, "age")
        assert pool.take_candidate("key", "adult", None) == {"age": 6}
        assert pool.take_candidate("key", "adult", 3) == {"age": 2}
        assert pool.take_candidate("key", "adult", 3) is None
        assert pool.take_candidate("missing", "adult", None) is None
        assert pool._age_distribution("key") == {4: 1}

    def test_rollback_restores_candidate_and_removes_household_id(self):
        pool = CandidatePool({"key": [{"age": 4}]}, "age")
        candidate = pool.take_candidate("key", "adult", None)
        candidate["household_id"] = 10
        pool.roll_back([("key", candidate, "household_id")])
        assert pool.take_candidate("key", "adult", None) == {"age": 4}


class TestGenerator:
    def test_household_sampling_is_reproducible_and_assigns_ids(self, generator):
        first = generator.simulate_household(3, random_seed=42)
        second = generator.simulate_household(3, random_seed=42)
        assert_frame_equal(first, second)
        assert first["household_id"].tolist() == [0, 1, 2]

    def test_evidence_batches_aggregate_demands(self, generator, blueprint):
        demands, valid = generator.generate_evidence_batches(blueprint)
        assert_frame_equal(valid, blueprint)
        assert demands == {
            PersonSimulationKey(0, 0, 0, 0, 5, 1): 2,
            PersonSimulationKey(0, 1, 0, 0, 1, None): 2,
            PersonSimulationKey(0, 2, 0, 0, 0, None): 2,
        }

    def test_oversized_households_are_filtered(self, generator, blueprint, caplog):
        blueprint.loc[1, ColumnNames.ADULTS] = 2
        valid = generator._check_blueprint(blueprint, max_rank=2)
        assert valid["household_id"].tolist() == [10]
        assert "exceed maximum supported size" in caplog.text

    @pytest.mark.parametrize(("include_kids", "expected_size"), [(True, 4), (False, 3)])
    def test_member_count_can_exclude_kids(self, generator, include_kids, expected_size):
        generator.p_data_include_kid = include_kids
        counts = {ColumnNames.ADULTS: 2, ColumnNames.MINORS: 1, ColumnNames.KIDS: 1}
        assert generator._count_member(counts) == (2, 1, 1, expected_size)

    @pytest.mark.parametrize(("rank", "expected"), [(0, "adult"), (1, "adult"), (2, "minor"), (3, "kid")])
    def test_required_group_by_rank(self, rank, expected):
        assert Generator._required_group(rank, adults=2, minors=1) == expected

    def test_person_sampling_applies_evidence_and_seed(self, generator):
        key = PersonSimulationKey(0, 0, 0, 0, 5, 1)
        args = (key, 2, 3, generator.p_model, generator.config, 42)
        returned_key, records = _batch_sample_persons(*args)
        assert returned_key == key
        assert len(records) == 6
        assert _batch_sample_persons(*args)[1] == records
        for person in records:
            assert person[ColumnNames.AGE_CATE] == 5
            assert person["gender"] == 1
            assert person[ColumnNames.MEMBER_RANK] == 0

    def test_successful_allocation_assigns_members_to_households(self, generator, blueprint):
        demands, _ = generator.generate_evidence_batches(blueprint)
        candidates = {
            key: [{ColumnNames.AGE_CATE: key.p_age} for _ in range(count)]
            for key, count in demands.items()
        }
        people, failed = generator.allocate_households(
            CandidatePool(candidates, ColumnNames.AGE_CATE), blueprint
        )
        assert failed == set()
        assert len(people) == 6
        for household_id in [10, 20]:
            members = [person for person in people if person["household_id"] == household_id]
            assert [person[ColumnNames.AGE_CATE] for person in members] == [5, 1, 0]

    def test_failed_allocation_rolls_back_partial_household(self, generator, blueprint):
        key = PersonSimulationKey(0, 0, 0, 0, 5, 1)
        pool = CandidatePool({key: [{ColumnNames.AGE_CATE: 5}]}, ColumnNames.AGE_CATE)
        people, failed = generator.allocate_households(pool, blueprint.iloc[:1])
        assert people == []
        assert failed == {10}
        assert pool.take_candidate(key, "adult", None) == {ColumnNames.AGE_CATE: 5}


class TestCalibration:
    def test_matrix_aligns_household_rows_and_marginal_columns(self, calibration_input):
        matrix = CalibrationMatrixBuilder.build(calibration_input)
        np.testing.assert_array_equal(matrix.x_h, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(matrix.x_p, [[1, 1], [0, 1]])
        np.testing.assert_array_equal(matrix.h_m_ar, [4, 2])
        np.testing.assert_array_equal(matrix.p_m_ar, [4, 6])
        np.testing.assert_array_equal(matrix.h_sizes, [2, 1])
        np.testing.assert_array_equal(matrix.w_initial, [3, 3])
        np.testing.assert_array_equal(matrix.X, [[1, 0, 1, 1], [0, 1, 0, 1]])
        np.testing.assert_array_equal(matrix.t, [4, 2, 4, 6])
        assert matrix.total_households == 6
        assert matrix.total_individuals == 10

    def test_categories_absent_from_sample_are_dropped(self, calibration_input):
        calibration_input.h_marginals["tenure"]["vacant"] = 1
        matrix = CalibrationMatrixBuilder.build(calibration_input)
        assert matrix.x_h.shape == (2, 2)
        np.testing.assert_array_equal(matrix.h_m_ar, [4, 2])

    def test_joint_categories_are_counted_per_household(self):
        df = pd.DataFrame({"id": [10, 10, 20], "a": ["x", "x", "y"], "b": [1, 2, 1]})
        counts, columns = CalibrationMatrixBuilder._sample_convert(
            df, {"a_b": {"x_1": 2, "x_2": 1, "y_1": 1}}, "id", [["a", "b"]]
        )
        assert columns == ["a_b_x_1", "a_b_x_2", "a_b_y_1"]
        np.testing.assert_array_equal(counts.to_numpy(), [[1, 1, 0], [0, 0, 1]])

    def test_marginal_array_preserves_column_order_and_integer_keys(self):
        result = CalibrationMatrixBuilder._marginal_to_array(
            {"age_cate": {0: 2, 1: 3}, "gender": {"F": 4}},
            ["gender_F", "age_cate_1", "age_cate_0"],
        )
        np.testing.assert_array_equal(result, [4, 3, 2])

    @pytest.mark.parametrize(
        ("calibrator_cls", "kwargs"),
        [(IPU, {}), (HIPF, {}), (GR, {"method": "linear"}),
         (GR, {"method": "raking"}), (GR, {"method": "logit"}),
         (CrossEntropy, {"alpha": 0})],
    )
    def test_calibrators_recover_known_weights(self, calibration_input, calibrator_cls, kwargs):
        matrix = CalibrationMatrixBuilder.build(calibration_input)
        result = calibrator_cls(matrix).fit(max_iter=3000, tolerance=1e-10, **kwargs)
        assert isinstance(result, CalibrationResult)
        assert result.is_converged
        assert np.isfinite(result.weights).all()
        assert (result.weights > 0).all()
        np.testing.assert_allclose(result.weights, [4, 2], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(matrix.X.T @ result.weights, matrix.t, rtol=1e-6)
        np.testing.assert_array_equal(matrix.w_initial, [3, 3])

    @pytest.mark.parametrize("calibrator_cls", [IPU, HIPF, GR])
    def test_zero_iterations_returns_unconverged_initial_weights(self, calibration_input, calibrator_cls):
        matrix = CalibrationMatrixBuilder.build(calibration_input)
        result = calibrator_cls(matrix).fit(max_iter=0)
        assert not result.is_converged
        assert result.iterations == 0
        np.testing.assert_array_equal(result.weights, matrix.w_initial)

    @pytest.mark.parametrize("bounds", [(1, 5), (2, 5), (0.1, 1), (0.1, 0.5)])
    def test_logit_rejects_bounds_that_exclude_one(self, calibration_input, bounds):
        calibrator = GR(CalibrationMatrixBuilder.build(calibration_input))
        with pytest.raises(ValueError, match="Lower bound must be < 1"):
            calibrator.fit(method="logit", bounds=bounds)

    def test_cross_entropy_gradient_matches_finite_differences(self, calibration_input):
        matrix = CalibrationMatrixBuilder.build(calibration_input)
        calibrator = CrossEntropy(matrix)
        multipliers = np.array([0.2, -0.1, 0.3, -0.2])
        args = (matrix.h_m_ar / 6, matrix.p_m_ar / 6, 0.01)
        _, gradient = calibrator._objective(multipliers, *args)
        numerical = np.empty_like(gradient)
        step = 1e-6
        for i in range(len(multipliers)):
            offset = np.zeros_like(multipliers)
            offset[i] = step
            numerical[i] = (
                calibrator._objective(multipliers + offset, *args)[0]
                - calibrator._objective(multipliers - offset, *args)[0]
            ) / (2 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)


class TestValidation:
    @pytest.mark.parametrize(
        ("metric", "expected"),
        [("root_mean_squared_error", 1), ("standardized_rmse", 0.25),
         ("total_abs_error", 2), ("relative_abs_error", 0.5),
         ("mean_abs_error", 1), ("r_square_pearson", 1),
         ("sum_square_modified_z_score", 4 / 3)],
    )
    def test_metrics_against_hand_calculated_values(self, metric, expected):
        # Estimated marginals [3, 5], observed marginals [2, 6].
        validation = Validation(np.array([[1, 2], [2, 3]]), np.array([2, 6]))
        assert getattr(validation, metric)() == pytest.approx(expected)

    def test_pearson_handles_constant_distributions(self):
        validation = Validation(np.array([[2, 2]]), np.array([1, 3]))
        assert validation.r_square_pearson() == 0

    @pytest.mark.parametrize("normalize", [True, False])
    def test_combination_srmse_aligns_missing_combinations_and_weights(self, normalize):
        origin = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": [1, 1, 1, 2]})
        synthetic = pd.DataFrame({"a": ["x", "y"], "b": [1, 2], "integer": [1, 3]})
        result, diff = Validation.combination_srmse(synthetic, origin, ["a", "b"], normalize)
        assert result == 1.0607
        scale = 0.25 if normalize else 1
        # Both attributes are encoded: original b values 1 and 2 become 0 and 1.
        assert diff.to_dict() == {(0, 0): scale, (1, 0): scale, (1, 1): -2 * scale}

    def test_combination_srmse_is_zero_for_identical_samples(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 1]})
        result, diff = Validation.combination_srmse(df, df, ["a", "b"])
        assert result == 0
        assert diff.eq(0).all()

    def test_compute_metrics_preserves_metadata(self):
        weights = np.array([[1, 2], [2, 3]])
        result = compute_metrics("ipu", weights.T, np.array([2, 6]), weights.T,
                                np.array([2, 6]), "households.csv", "people.csv", True)
        assert result["ipu"]["filename_h"] == "households.csv"
        assert result["ipu"]["filename_p"] == "people.csv"
        assert result["ipu"]["is_converged"] is True
        assert result["ipu"]["h_rmse"] == pytest.approx(1)
        assert result["ipu"]["p_rmse"] == pytest.approx(1)
        assert result["ipu"]["h_zscore"] == pytest.approx(4 / 3)
        assert result["ipu"]["p_zscore"] == pytest.approx(4 / 3)
class TestConfigAndIO:
    def test_config_load_creates_directories_and_resolves_interpolation(self, tmp_path):
        # Load a temporary copy; the repository config's output paths are not used.
        source = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        config = OmegaConf.load(source)
        config.paths.output_dir = str(tmp_path / "output")
        config.calibration.kwargs.bounds = [0.001, 50.0]
        config_path = tmp_path / "config.yaml"
        OmegaConf.save(config, config_path)
        result = AppConfig.load(str(config_path))
        assert isinstance(result, AppConfig)
        assert isinstance(result.train, TrainConfig)
        assert isinstance(result.calibration, CalibrationConfig)
        assert Path(result.paths.output_dir).is_dir()
        assert Path(result.paths.dags_dir).is_dir()
        assert Path(result.paths.bundle_file) == tmp_path / "output" / "bn_bundle.pkl"
        assert result.synthesis.random_seed == 42

    def test_config_rejects_unknown_fields(self, tmp_path):
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("unknown_field: true\n", encoding="utf-8")
        with pytest.raises(ConfigKeyError):
            AppConfig.load(str(config_path))

    def test_calibration_kwargs_defaults_are_independent(self):
        args = (100, 1e-6, {}, {}, [], [])
        first, second = CalibrationConfig(*args), CalibrationConfig(*args)
        first.kwargs["alpha"] = 0.01
        assert second.kwargs == {}

    def test_bundle_round_trip_preserves_models_and_schema(self, generator, tmp_path):
        path = tmp_path / "bundle.pkl"
        save_bn_bundle(generator.h_model, generator.p_model, generator.config, str(path))
        result = load_bn_bundle(str(path))
        assert result["config"] == generator.config
        for key in ["h_model", "p_model"]:
            assert result[key].check_model()
            original = getattr(generator, key)
            assert set(result[key].nodes()) == set(original.nodes())
            for cpd in original.get_cpds():
                np.testing.assert_array_equal(result[key].get_cpds(cpd.variable).values, cpd.values)

    def test_load_bundle_rejects_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bn_bundle(str(tmp_path / "missing.pkl"))

    def test_csv_creates_directories_and_avoids_overwriting(self, tmp_path):
        directory = tmp_path / "nested" / "output"
        first = pd.DataFrame({"household_id": [1], "age": [40]})
        second = pd.DataFrame({"household_id": [2], "age": [50]})
        save_csv(first, directory, "people.csv", index=False)
        save_csv(second, directory, "people", index=False)
        assert_frame_equal(pd.read_csv(directory / "people.csv"), first)
        assert_frame_equal(pd.read_csv(directory / "people_1.csv"), second)


class TestSeeding:
    @pytest.mark.parametrize("seed", [0, 1, 42, 2**32 - 1])
    def test_seed_is_reproducible_and_in_uint32_range(self, seed):
        result = random_seeds(seed)
        assert isinstance(result, int)
        assert 0 <= result < np.iinfo(np.uint32).max
        assert result == random_seeds(seed)

    def test_none_preserves_unseeded_behavior(self):
        assert random_seeds(None) is None
