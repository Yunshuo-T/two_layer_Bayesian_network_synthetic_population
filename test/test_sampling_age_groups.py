"""Age-group sampling tests with small, fitted Bayesian networks."""

from collections import Counter
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from synpop.schema import SchemaConstants, ModelSchema
from synpop.synthesis import generator as sampling
from synpop.synthesis.evidence import PersonSimulationKey


@pytest.fixture
def age_group_schema():
    code_to_attr = {
        SchemaConstants.HEAD_AGE: {0: 3},
        SchemaConstants.AGE_CATE: {0: 1, 1: 2, 2: 3},
        SchemaConstants.HEAD_GENDER: {0: "F"},
        "gender": {0: "M", 1: "F"},
    }
    attr_to_code = {
        attribute: {value: code for code, value in values.items()}
        for attribute, values in code_to_attr.items()
    }
    return ModelSchema(
        code_to_attr=code_to_attr,
        attr_to_code=attr_to_code,
        gender_col="gender",
        h_id_col="household_id",
        age_states_by_group={"adult": [2], "minor": [1, 0, 1], "kid": []},
    )


def _independent_model(states, probabilities=None):
    probabilities = probabilities or {}
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(states)
    for variable, values in states.items():
        weights = probabilities.get(variable, np.full(len(values), 1 / len(values)))
        model.add_cpds(
            TabularCPD(
                variable=variable,
                variable_card=len(values),
                values=np.asarray(weights, dtype=float).reshape(-1, 1),
                state_names={variable: values},
            )
        )
    assert model.check_model()
    return model


@pytest.fixture
def age_group_models():
    household_model = _independent_model(
        {
            SchemaConstants.H_TYPE: [0],
            SchemaConstants.HEAD_AGE: [0],
            SchemaConstants.HEAD_GENDER: [0],
            SchemaConstants.ADULTS: [1],
            SchemaConstants.MINORS: [1],
            SchemaConstants.KIDS: [1],
        }
    )
    person_model = _independent_model(
        {
            SchemaConstants.H_TYPE: [0],
            SchemaConstants.HEAD_AGE: [0],
            SchemaConstants.HEAD_GENDER: [0],
            SchemaConstants.MEMBER_RANK: [0, 1],
            SchemaConstants.AGE_CATE: [0, 1, 2],
            "gender": [0, 1],
        },
        {SchemaConstants.AGE_CATE: [0.2, 0.6, 0.2]},
    )
    return household_model, person_model


@pytest.fixture
def age_group_generator(age_group_models, age_group_schema):
    household_model, person_model = age_group_models
    generator = sampling.Generator(
        household_model, person_model, age_group_schema, p_data_include_kid=False
    )
    # Threshold filtering is outside these age-sampling regression tests.
    generator.probabilities = {}
    generator.probability_attributes = []
    return generator


def _key(group, rank=1):
    return PersonSimulationKey.make_evidence_key(
        h_type=0,
        rank=rank,
        head_age=0,
        head_gender=0,
        required_group=group,
        head_age_to_person_age={0: 2},
        head_gender_to_person_gender={0: 1},
        age_states_by_group={"adult": [2], "minor": [1, 0, 1], "kid": []},
    )


def test_evidence_keys_keep_exact_head_age_and_canonical_member_states():
    head = _key("adult", rank=0)
    minor = _key("minor")

    assert head == PersonSimulationKey(0, 0, 0, 0, (2,), 1)
    assert minor == PersonSimulationKey(0, 1, 0, 0, (0, 1), None)
    assert {minor: 2}[PersonSimulationKey(0, 1, 0, 0, (0, 1), None)] == 2


def test_evidence_key_rejects_group_without_supported_age_states():
    with pytest.raises(ValueError, match="No supported age states.*kid"):
        _key("kid")


def test_member_sampling_preserves_weighted_age_distribution_and_seed(
    age_group_models, age_group_schema
):
    _, person_model = age_group_models
    key = _key("minor")
    args = (key, 1000, 4, person_model, age_group_schema, 42)

    returned_key, records = sampling._batch_sample_persons(*args)

    assert returned_key == key
    assert 0 < len(records) < 4000
    assert sampling._batch_sample_persons(*args)[1] == records
    counts = Counter(person[SchemaConstants.AGE_CATE] for person in records)
    assert set(counts) == {0, 1}
    assert counts[1] / len(records) == pytest.approx(0.75, abs=0.05)
    assert {person["gender"] for person in records} == {0, 1}
    assert {person[SchemaConstants.MEMBER_RANK] for person in records} == {1}


def test_head_sampling_clamps_scalar_age_and_gender_for_full_proposal_budget(
    age_group_models, age_group_schema
):
    _, person_model = age_group_models
    key = _key("adult", rank=0)

    returned_key, records = sampling._batch_sample_persons(
        key, 3, 4, person_model, age_group_schema, 42
    )

    assert returned_key == key
    assert len(records) == 12
    assert {person[SchemaConstants.AGE_CATE] for person in records} == {2}
    assert {person["gender"] for person in records} == {1}
    assert {person[SchemaConstants.MEMBER_RANK] for person in records} == {0}


def test_rejected_age_proposals_return_empty_without_refilling(
    monkeypatch, age_group_schema
):
    sampler = Mock()
    sampler.forward_sample.return_value = pd.DataFrame(
        {SchemaConstants.AGE_CATE: [2] * 6}
    )
    monkeypatch.setattr(sampling, "BayesianModelSampling", Mock(return_value=sampler))
    key = _key("minor")

    assert sampling._batch_sample_persons(key, 2, 3, None, age_group_schema, 17) == (
        key,
        [],
    )
    sampler.forward_sample.assert_called_once()
    arguments = sampler.forward_sample.call_args.kwargs
    assert arguments["size"] == 6
    assert arguments["seed"] == 17
    assert SchemaConstants.AGE_CATE not in arguments["partial_samples"].columns


@pytest.mark.parametrize(("count", "multiplier"), [(0, 4), (3, 0)])
def test_zero_proposal_budget_returns_empty_without_constructing_sampler(
    monkeypatch, age_group_schema, count, multiplier
):
    sampler_factory = Mock(side_effect=AssertionError("Sampling must not start."))
    monkeypatch.setattr(sampling, "BayesianModelSampling", sampler_factory)
    key = _key("minor")

    assert sampling._batch_sample_persons(
        key, count, multiplier, None, age_group_schema, 17
    ) == (key, [])
    sampler_factory.assert_not_called()


def test_worker_rejects_direct_key_without_age_states(monkeypatch, age_group_schema):
    sampler_factory = Mock(side_effect=AssertionError("Sampling must not start."))
    monkeypatch.setattr(sampling, "BayesianModelSampling", sampler_factory)
    key = PersonSimulationKey(0, 1, 0, 0, (), None)

    with pytest.raises(ValueError, match="at least one age state"):
        sampling._batch_sample_persons(key, 2, 3, None, age_group_schema, 17)
    sampler_factory.assert_not_called()


def test_synthesis_uses_multiple_minor_states_with_process_workers(age_group_generator):
    households, people = age_group_generator.synthesize(
        total_population=120,
        num_households=60,
        pool_multiplier=8,
        max_retries=1,
        random_seed=42,
        max_workers=2,
        threshold=0,
    )

    assert len(households) == 60
    assert len(people) == 120
    assert people.groupby("household_id").size().eq(2).all()
    assert set(people["household_id"]) == set(households["household_id"])
    heads = people.loc[people[SchemaConstants.MEMBER_RANK].eq(0)]
    minors = people.loc[people[SchemaConstants.MEMBER_RANK].eq(1)]
    assert len(heads) == len(minors) == 60
    assert set(heads[SchemaConstants.AGE_CATE]) == {2}
    assert set(heads["gender"]) == {1}
    assert set(minors[SchemaConstants.AGE_CATE]) == {0, 1}
    assert set(people[SchemaConstants.MEMBER_RANK]) == {0, 1}
    assert households[SchemaConstants.KIDS].eq(1).all()
    assert (
        households[SchemaConstants.ADULTS] + households[SchemaConstants.MINORS]
    ).eq(2).all()
