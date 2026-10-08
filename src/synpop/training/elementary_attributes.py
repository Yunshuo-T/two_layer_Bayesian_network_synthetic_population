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
    p_df: pd.DataFrame,
    h_id_col: str = "H_ID",
    h_df: pd.DataFrame | None = None,
    composition_cols: dict[str,str] | None = None,
) -> pd.DataFrame:
    """Use household counts when available in household dataframe, otherwise count person age categories in person dataframe.

    ``composition_cols`` maps ``adults``, ``minors`` and ``kids`` to their
    household source columns.
    """
    
    indicators = pd.DataFrame(
        {
            h_id_col: p_df[h_id_col],
            ColumnNames.ADULTS: (p_df[ColumnNames.AGE_CATE] >= 2).astype(int),
            ColumnNames.MINORS: (p_df[ColumnNames.AGE_CATE] == 1).astype(int),
            ColumnNames.KIDS: (p_df[ColumnNames.AGE_CATE] == 0).astype(int),
        }
    )
    
    composition = indicators.groupby(h_id_col, observed=True).sum().reset_index()
    
    if composition_cols:
        if h_df is None:
            raise ValueError("The household dataframe cannot be None when composition cols are supplied.")
        else:
            household_counts = h_df[
                [h_id_col, *composition_cols.values()]
            ].rename(
                columns={
                    source: target
                    for target, source in composition_cols.items()
                }
            )

            composition = composition.drop(
                columns=list(composition_cols)
            ).merge(
                household_counts,
                on=h_id_col,
                how="left",
                validate="one_to_one",
            )

    return composition 


def _add_household_type(
    p_df: pd.DataFrame,
    h_id_col: str = "H_ID",
    h_df: pd.DataFrame | None = None,
    composition_cols: dict[str, str] | None = None,
) -> pd.DataFrame:

    # Aggregate counts per household
    comp = _calculate_household_composition(
        p_df, h_id_col=h_id_col, h_df=h_df, composition_cols=composition_cols
    )
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
    return p_df.drop(columns=merge_cols[1:], errors="ignore").merge(
        comp[merge_cols], how="left", on=h_id_col, validate="many_to_one"
    )


def _sync_household_attributes(
    h_df: pd.DataFrame,
    p_df: pd.DataFrame,
    h_id_col: str,
) -> pd.DataFrame:
    """Extracts household-level attributes from p_df and merges them into h_df."""
    cols = [c for c in ColumnNames.HOUSEHOLD_COLS]
    h_features = p_df[[h_id_col] + cols].drop_duplicates(subset=[h_id_col])
    return h_df.drop(columns=cols, errors="ignore").merge(
        h_features, on=h_id_col, how="left", validate="many_to_one"
    )


def add_elementary_attributes(
    raw_h_df: pd.DataFrame,
    raw_p_df: pd.DataFrame,
    age_col: str,
    gender_col: str,
    h_id_col: str,
    age_bin: list | None = None,
    composition_cols: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive training features, preferring household composition counts.

    For custom count column names, supply a mapping such as
    ``{"adults": "H_persons_1899", "minors": "H_persons_0617",
    "kids": "H_persons_0005"}``. Canonical names are detected automatically.
    """

    processed_p_df = (
        raw_p_df.pipe(_add_age_category, age_col=age_col, bin=age_bin)
        .pipe(_add_member_rank, h_id_col=h_id_col, age_col=age_col)
        .pipe(_add_head_attributes, h_id_col=h_id_col, gender_col=gender_col)
        .pipe(
            _add_household_type,
            h_id_col=h_id_col,
            h_df=raw_h_df,
            composition_cols=composition_cols,
        )
    )

    processed_h_df = _sync_household_attributes(
        h_df=raw_h_df,
        p_df=processed_p_df,
        h_id_col=h_id_col,
    )

    return processed_h_df, processed_p_df
