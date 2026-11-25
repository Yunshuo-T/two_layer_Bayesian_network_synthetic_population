from synpop.schema import SchemaConstants
from synpop.synthesis.candidate import CandidatePool


AGE = SchemaConstants.AGE_CATE
AGE_GROUPS = {"kid": [], "minor": [0, 1], "adult": [2, 3]}


def _take(
    pool,
    *,
    required_group="minor",
    previous_age=None,
    age_groups=None,
    include_kids=False,
    threshold=0.0,
    probabilities=None,
):
    return pool.take_candidate(
        h_dict={},
        threshold=threshold,
        total_population=100,
        evidence_key="key",
        required_group=required_group,
        previous_age=previous_age,
        probabilities=(
            {(state,): 1.0 for state in range(4)}
            if probabilities is None
            else probabilities
        ),
        attributes=[AGE],
        age_states_by_group=AGE_GROUPS if age_groups is None else age_groups,
        p_data_include_kid=include_kids,
    )


def test_both_minor_states_are_selected_in_sampled_order():
    records = [{AGE: 0}, {AGE: 1}]
    pool = CandidatePool({"key": records}, AGE)

    younger = _take(pool)
    older = _take(pool)

    assert younger == {AGE: 0}
    assert older == {AGE: 1}
    assert _take(pool) is None
    assert records == [{AGE: 0}, {AGE: 1}]


def test_ascending_household_age_is_rejected_without_consuming_candidate():
    pool = CandidatePool({"key": [{AGE: 1}, {AGE: 0}]}, AGE)

    assert _take(pool, previous_age=0) == {AGE: 0}
    assert _take(pool, previous_age=0) is None
    assert _take(pool) == {AGE: 1}


def test_kid_candidates_are_excluded_when_kids_are_disabled():
    groups = {"kid": [0], "minor": [1, 2], "adult": [3]}
    pool = CandidatePool({"key": [{AGE: 0}]}, AGE)

    assert _take(pool, required_group="kid", age_groups=groups) is None
    assert _take(
        pool, required_group="kid", age_groups=groups, include_kids=True
    ) == {AGE: 0}


def test_probability_threshold_keeps_rejected_candidate_available():
    pool = CandidatePool({"key": [{AGE: 0}, {AGE: 1}]}, AGE)
    probabilities = {(0,): 0.005, (1,): 0.01}

    assert _take(pool, threshold=1.0, probabilities=probabilities) == {AGE: 1}
    assert _take(pool, threshold=0.0, probabilities=probabilities) == {AGE: 0}


def test_rollback_restores_candidate_and_removes_household_id():
    pool = CandidatePool({"key": [{AGE: 1}]}, AGE)
    candidate = _take(pool)
    candidate["household_id"] = 42

    pool.roll_back([("key", candidate, "household_id")])

    restored = _take(pool)
    assert restored is candidate
    assert restored == {AGE: 1}
