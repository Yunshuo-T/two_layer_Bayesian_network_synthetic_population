import pandas as pd
import numpy as np
from src.core.constants import ColumnNames


def classify_age(
    p_df: pd.DataFrame,
    age_col: str,
    bins: list | None = None,
) -> pd.DataFrame:

    if bins is None:
        bins = [-1, 5, 17, 29, 39, 49, 59, 69, 79, float("inf")]
    labels = list(range(len(bins) - 1))
    p_df[age_col] = p_df[age_col].astype(int)
    p_df[ColumnNames.AGE_CATE] = pd.cut(
        p_df[age_col],
        bins=bins,
        labels=labels,
    ).astype(int)

    return p_df


def add_member_rank(
    p_df: pd.DataFrame,
    h_id_col: str,
    age_col: str,
) -> pd.DataFrame:

    p_df = p_df.sort_values(
        [h_id_col, age_col],
        ascending=[True, False],
    ).copy()
    p_df[ColumnNames.MEMBER_RANK] = p_df.groupby(h_id_col, observed=True).cumcount()
    return p_df


def add_head_attributes(
    p_df: pd.DataFrame,
    h_id_col: str,
    gender_col: str,
) -> pd.DataFrame:

    p_df = p_df.copy()
    grouped = p_df.groupby(h_id_col)
    p_df[ColumnNames.HEAD_AGE] = grouped[ColumnNames.AGE_CATE].transform("first")
    p_df[ColumnNames.HEAD_GENDER] = grouped[gender_col].transform("first")
    return p_df


def calculate_household_composition(
    p_df: pd.DataFrame,
    h_id_col: str = "H_ID",
) -> pd.DataFrame:

    indicators = pd.DataFrame(
        {
            h_id_col: p_df[h_id_col],
            ColumnNames.ADULTS: (p_df[ColumnNames.AGE_CATE] >= 2).astype(int),
            ColumnNames.MINORS: (p_df[ColumnNames.AGE_CATE] == 1).astype(int),
            ColumnNames.KIDS: (p_df[ColumnNames.AGE_CATE] == 0).astype(int),
        }
    )

    return indicators.groupby(h_id_col, observed=True).sum().reset_index()


def classify_household_type(
    p_df: pd.DataFrame,
    h_id_col: str = "H_ID",
) -> pd.DataFrame:

    # Aggregate counts per household
    comp = calculate_household_composition(p_df, h_id_col=h_id_col)
    # Determine the Adult Prefix
    prefix = np.select(
        [
            comp[ColumnNames.ADULTS] == 1,
            comp[ColumnNames.ADULTS] == 2,
            comp[ColumnNames.ADULTS] > 2,
        ],
        ["single", "couple", "only_adults"],
        default="other",
    )
    # Determine the Child Suffix
    has_kids = comp[ColumnNames.KIDS] > 0
    has_minors = comp[ColumnNames.MINORS] > 0
    suffix = np.select(
        [has_kids & has_minors, has_kids, has_minors],
        ["_with_children", "_with_kid", "_with_minor"],
        default="",
    )
    comp[ColumnNames.H_TYPE] = prefix + suffix
    # change 'adults' to 'only_adults'
    merge_cols = [
        h_id_col,
        ColumnNames.H_TYPE,
        ColumnNames.ADULTS,
        ColumnNames.MINORS,
        ColumnNames.KIDS,
    ]
    return p_df.merge(comp[merge_cols], how="left", on=h_id_col)


def sync_household_attributes(
    h_df: pd.DataFrame,
    p_df: pd.DataFrame,
    h_id_col: str = "H_ID",
) -> pd.DataFrame:
    """Extracts derived household-level attributes from p_df and merges them into h_df."""
    cols = [c for c in ColumnNames.HOUSEHOLD_COLS]
    h_features = p_df[[h_id_col] + cols].drop_duplicates(subset=[h_id_col])
    return h_df.merge(h_features, on=h_id_col, how="left")


def add_elementary_attributes(
    raw_h_df: pd.DataFrame,
    raw_p_df: pd.DataFrame,
    age_col: str = "AGE",
    gender_col: str = "SEXE",
    h_id_col: str = "H_ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    processed_p_df = (
        raw_p_df.pipe(classify_age, age_col=age_col)
        .pipe(add_member_rank, h_id_col=h_id_col, age_col=age_col)
        .pipe(add_head_attributes, h_id_col=h_id_col, gender_col=gender_col)
        .pipe(classify_household_type, h_id_col=h_id_col)
    )

    processed_h_df = sync_household_attributes(
        h_df=raw_h_df,
        p_df=processed_p_df,
        h_id_col=h_id_col,
    )

    return processed_h_df, processed_p_df
