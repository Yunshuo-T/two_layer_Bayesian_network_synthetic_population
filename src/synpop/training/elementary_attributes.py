import pandas as pd
from pandas.api.types import is_string_dtype, is_object_dtype
import numpy as np
from synpop.schema import ColumnNames


def _add_age_category(
    p_df: pd.DataFrame,
    age_col: str,
    bin: list | None = None,
) -> pd.DataFrame:
    """
    Classify the age of household members into age categories.
    The age information of the original dataframe can be integers or existing 
    age categories. If you want to use existing age categories, you still need to 
    give the age category bin. 
    Args:
        p_df (pd.DataFrame): The dataframe of household members.
        age_col (str): The column name indicating the member age.
        bin (list | None, optional): The target age categories for classifying the age of household members. Default: bins = [-1, 5, 17, 29, 39, 49, 59, 69, 79, float("inf")]

    Returns:
        pd.DataFrame: _description_
    """
    df = p_df.copy()
    if bin is None:
        bin = [-1, 5, 17, 29, 39, 49, 59, 69, 79, float("inf")]
    labels = list(range(len(bin) - 1))
    # if the age info is existing category, extract the left age from the interval.
    if is_string_dtype(df[age_col]) or is_object_dtype(df[age_col]):
        df[age_col] = df[age_col].astype(str).str.extract(r'(\d+)')[0]
    df[age_col] = df[age_col].astype(int)
    df[ColumnNames.AGE_CATE] = pd.cut(
        df[age_col],
        bins=bin,
        labels=labels,
    ).astype(int)

    return df


def _add_member_rank(
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


def _add_head_attributes(
    p_df: pd.DataFrame,
    h_id_col: str,
    gender_col: str,
) -> pd.DataFrame:

    p_df = p_df.copy()
    grouped = p_df.groupby(h_id_col)
    p_df[ColumnNames.HEAD_AGE] = grouped[ColumnNames.AGE_CATE].transform("first")
    p_df[ColumnNames.HEAD_GENDER] = grouped[gender_col].transform("first")
    return p_df


def _calculate_household_composition(
    # TODO consider using the household dataframe to classify the household composition, to prevent the situation the the household member data excludes children.
    # TODO Can ask the p_df inherit the information from h_df, specify the composition cols. 
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


def _add_household_type(
    p_df: pd.DataFrame,
    h_id_col: str = "H_ID",
) -> pd.DataFrame:

    # Aggregate counts per household
    comp = _calculate_household_composition(p_df, h_id_col=h_id_col)
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


def _sync_household_attributes(
    h_df: pd.DataFrame,
    p_df: pd.DataFrame,
    h_id_col: str,
) -> pd.DataFrame:
    """Extracts household-level attributes from p_df and merges them into h_df."""
    cols = [c for c in ColumnNames.HOUSEHOLD_COLS]
    h_features = p_df[[h_id_col] + cols].drop_duplicates(subset=[h_id_col])
    return h_df.merge(h_features, on=h_id_col, how="left")


def add_elementary_attributes(
    raw_h_df: pd.DataFrame,
    raw_p_df: pd.DataFrame,
    age_col: str,
    gender_col: str,
    h_id_col: str,
    age_bin:list|None = None,
    
) -> tuple[pd.DataFrame, pd.DataFrame]:

    processed_p_df = (
        raw_p_df.pipe(_add_age_category, age_col=age_col,bin=age_bin)
        .pipe(_add_member_rank, h_id_col=h_id_col, age_col=age_col)
        .pipe(_add_head_attributes, h_id_col=h_id_col, gender_col=gender_col)
        .pipe(_add_household_type, h_id_col=h_id_col)
    )

    processed_h_df = _sync_household_attributes(
        h_df=raw_h_df,
        p_df=processed_p_df,
        h_id_col=h_id_col,
    )

    return processed_h_df, processed_p_df
