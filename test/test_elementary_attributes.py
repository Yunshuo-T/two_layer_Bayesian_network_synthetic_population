import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from synpop.schema import ColumnNames
from synpop.training.elementary_attributes import (
    _add_age_category,
    _add_member_rank,
    _add_head_attributes,
    _calculate_household_composition,
    _add_household_type,
    add_elementary_attributes,
)


H_ID = "household_id"
AGE = "age"
GENDER = "gender"

COMPOSITION_COLS = {
    ColumnNames.ADULTS: "adult_count",
    ColumnNames.MINORS: "minor_count",
    ColumnNames.KIDS: "kid_count",
}


@pytest.fixture
def sample_data():
    households = pd.DataFrame({
        H_ID: [30, 10, 20],
    })

    people = pd.DataFrame({
        H_ID: [20, 10, 20, 30, 20, 30],
        AGE: [4, 50, 40, 35, 12, 32],
        GENDER: ["M", "F", "F", "M", "F", "F"],
    })

    return households, people


@pytest.fixture
def composition_data():
    households = pd.DataFrame({
        H_ID: [10, 20],
        "adult_count": [1, 2],
        "minor_count": [0, 1],
        "kid_count": [2, 3],
    })

    people = pd.DataFrame({
        H_ID: [10, 20, 20],
        ColumnNames.AGE_CATE: [2, 2, 2],
    })

    return households, people


class TestAgeCategory:

    @pytest.mark.parametrize(
        ("age", "expected"),
        [
            (0, 0), (5, 0),
            (6, 1), (17, 1),
            (18, 2), (29, 2),
            (30, 3), (39, 3),
            (40, 4), (49, 4),
            (50, 5), (59, 5),
            (60, 6), (69, 6),
            (70, 7), (79, 7),
            (80, 8), (105, 8),
        ],
    )
    @pytest.mark.parametrize("as_string", [False, True])
    def test_default_bins(self, age, expected, as_string: bool):
        value = str(age) if as_string else age
        df = pd.DataFrame({AGE: [value]})

        result = _add_age_category(df, AGE)

        assert result[ColumnNames.AGE_CATE].iloc[0] == expected

    def test_custom_bins(self):
        df = pd.DataFrame({AGE: [0, 10, 11, 20]})

        result = _add_age_category(
            df, AGE, bin=[-1, 10, 20]
        )

        assert result[ColumnNames.AGE_CATE].tolist() == [
            0, 0, 1, 1
        ]

    def test_existing_age_intervals(self):
        df = pd.DataFrame({
            AGE: ["0-5", "6-17", "18-29", "30-39"]
        })

        result = _add_age_category(df, AGE)

        assert result[ColumnNames.AGE_CATE].tolist() == [
            0, 1, 2, 3
        ]

    def test_input_not_modified(self):
        df = pd.DataFrame({AGE: ["10", "20"]})
        original = df.copy(deep=True)

        _add_age_category(df, AGE)

        assert_frame_equal(df, original)


class TestMemberAttributes:

    def test_member_rank(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]):
        _, people = sample_data

        result = _add_member_rank(people, H_ID, AGE)

        expected = {
            10: [0],
            20: [0, 1, 2],
            30: [0, 1],
        }

        for h_id, ranks in expected.items():
            actual = result.loc[
                result[H_ID] == h_id,
                ColumnNames.MEMBER_RANK,
            ].tolist()
            assert actual == ranks

        assert result.loc[
            result[H_ID] == 20, AGE
        ].tolist() == [40, 12, 4]

    def test_head_attributes(self):
        people = pd.DataFrame({
            H_ID: [10, 10, 20, 20],
            ColumnNames.AGE_CATE: [5, 2, 4, 1],
            GENDER: ["F", "M", "M", "F"],
        })

        result = _add_head_attributes(
            people, H_ID, GENDER
        )

        assert result[ColumnNames.HEAD_AGE].tolist() == [
            5, 5, 4, 4
        ]
        assert result[ColumnNames.HEAD_GENDER].tolist() == [
            "F", "F", "M", "M"
        ]


class TestHouseholdComposition:

    @pytest.mark.parametrize(
        ("ages", "expected"),
        [
            ([2], (1, 0, 0)),
            ([2, 3], (2, 0, 0)),
            ([2, 1, 0], (1, 1, 1)),
            ([0, 1], (0, 1, 1)),
            ([2, 2, 0, 0], (2, 0, 2)),
        ],
    )
    def test_inferred_counts(self, ages, expected):
        people = pd.DataFrame({
            H_ID: [10] * len(ages),
            ColumnNames.AGE_CATE: ages,
        })

        result = _calculate_household_composition(
            people, H_ID
        )

        actual = tuple(
            result.loc[0, col]
            for col in (
                ColumnNames.ADULTS,
                ColumnNames.MINORS,
                ColumnNames.KIDS,
            )
        )

        assert actual == expected

    def test_household_counts_override_person_counts(
        self, composition_data: tuple[pd.DataFrame, pd.DataFrame]
    ):
        households, people = composition_data

        result = _calculate_household_composition(
            people,
            h_id_col=H_ID,
            h_df=households,
            composition_cols=COMPOSITION_COLS,
        )

        expected = pd.DataFrame({
            H_ID: [10, 20],
            ColumnNames.ADULTS: [1, 2],
            ColumnNames.MINORS: [0, 1],
            ColumnNames.KIDS: [2, 3],
        })

        assert_frame_equal(result, expected)

    def test_mapping_requires_household_dataframe(
        self, composition_data: tuple[pd.DataFrame, pd.DataFrame]
    ):
        _, people = composition_data

        with pytest.raises(ValueError, match="cannot be None"):
            _calculate_household_composition(
                people,
                H_ID,
                composition_cols=COMPOSITION_COLS,
            )

    def test_input_not_modified(self, composition_data: tuple[pd.DataFrame, pd.DataFrame]):
        households, people = composition_data

        original_h = households.copy(deep=True)
        original_p = people.copy(deep=True)

        _calculate_household_composition(
            people,
            H_ID,
            h_df=households,
            composition_cols=COMPOSITION_COLS,
        )

        assert_frame_equal(households, original_h)
        assert_frame_equal(people, original_p)


class TestHouseholdType:

    @pytest.mark.parametrize(
        ("ages", "expected"),
        [
            ([2], "single"),
            ([2, 3], "couple"),
            ([2, 3, 4], "only_adults"),
            ([2, 0], "single_with_kid"),
            ([2, 1], "single_with_minor"),
            ([2, 1, 0], "single_with_children"),
            ([0, 1], "other_with_children"),
            ([2, 2, 0, 1], "couple_with_children"),
            ([2, 2, 0], "couple_with_kid"),
        ],
    )
    def test_household_types(self, ages, expected):
        people = pd.DataFrame({
            H_ID: [10] * len(ages),
            ColumnNames.AGE_CATE: ages,
        })

        result = _add_household_type(people, H_ID)

        assert result[ColumnNames.H_TYPE].eq(expected).all()

    def test_replaces_existing_features(self):
        people = pd.DataFrame({
            H_ID: [10],
            ColumnNames.AGE_CATE: [2],
            ColumnNames.H_TYPE: ["stale"],
            ColumnNames.ADULTS: [99],
        })

        result = _add_household_type(people, H_ID)

        assert result[ColumnNames.H_TYPE].iloc[0] == "single"
        assert result[ColumnNames.ADULTS].iloc[0] == 1
        assert not any(
            col.endswith(("_x", "_y"))
            for col in result.columns
        )


class TestElementaryAttributesIntegration:

    def test_full_pipeline(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]):
        households, people = sample_data

        processed_h, processed_p = add_elementary_attributes(
            households,
            people,
            age_col=AGE,
            gender_col=GENDER,
            h_id_col=H_ID,
        )

        person_20 = processed_p.loc[
            processed_p[H_ID] == 20
        ]

        assert person_20[AGE].tolist() == [40, 12, 4]
        assert person_20[ColumnNames.MEMBER_RANK].tolist() == [
            0, 1, 2
        ]
        assert person_20[ColumnNames.HEAD_AGE].tolist() == [
            4, 4, 4
        ]
        assert person_20[ColumnNames.HEAD_GENDER].tolist() == [
            "F", "F", "F"
        ]

        household_20 = processed_h.set_index(H_ID).loc[20]

        assert household_20[ColumnNames.H_TYPE] == (
            "single_with_children"
        ) # type: ignore

        assert [
            household_20[col]
            for col in (
                ColumnNames.ADULTS,
                ColumnNames.MINORS,
                ColumnNames.KIDS,
            )
        ] == [1, 1, 1]

        assert processed_h[H_ID].tolist() == [30, 10, 20]

    def test_pipeline_with_household_counts(
        self, composition_data: tuple[pd.DataFrame, pd.DataFrame]
    ):
        households, _ = composition_data

        people = pd.DataFrame({
            H_ID: [10, 20, 20],
            AGE: [50, 30, 40],
            GENDER: ["F", "M", "F"],
        })

        processed_h, processed_p = add_elementary_attributes(
            households,
            people,
            AGE,
            GENDER,
            H_ID,
            composition_cols=COMPOSITION_COLS,
        )

        expected_types = {
            10: "single_with_kid",
            20: "couple_with_children",
        }

        for h_id, expected_type in expected_types.items():
            assert processed_h.loc[
                processed_h[H_ID] == h_id,
                ColumnNames.H_TYPE,
            ].iloc[0] == expected_type

            assert processed_p.loc[
                processed_p[H_ID] == h_id,
                ColumnNames.H_TYPE,
            ].eq(expected_type).all()

    def test_inputs_not_modified(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]):
        households, people = sample_data

        original_h = households.copy(deep=True)
        original_p = people.copy(deep=True)

        add_elementary_attributes(
            households, people, AGE, GENDER, H_ID
        )

        assert_frame_equal(households, original_h)
        assert_frame_equal(people, original_p)

    def test_household_features_synchronized(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]):
        households, people = sample_data

        processed_h, processed_p = add_elementary_attributes(
            households, people, AGE, GENDER, H_ID
        )

        features = list(ColumnNames.HOUSEHOLD_COLS)

        person_features = (
            processed_p[[H_ID, *features]]
            .drop_duplicates(subset=H_ID)
            .sort_values(H_ID)
            .reset_index(drop=True)
        )

        household_features = (
            processed_h[[H_ID, *features]]
            .sort_values(H_ID)
            .reset_index(drop=True)
        )

        assert_frame_equal(
            person_features,
            household_features,
        )
