import pandas as pd
from synpop.schema import GlobleVariables

def cate_codes(df: pd.DataFrame, attributes: list, explicit_orders: dict = None):  # type: ignore
    df = df.copy()
    attr_to_code = {}
    code_to_attr = {}

    if explicit_orders is None:
        explicit_orders = {}

    for attr in attributes:
        if attr in explicit_orders:
            cat_type = pd.CategoricalDtype(
                categories=explicit_orders[attr], ordered=True
            )
            df[attr] = df[attr].astype(cat_type)
        else:
            df[attr] = df[attr].astype("category")

        categories = df[attr].cat.categories
        attr_to_code[attr] = {val: i for i, val in enumerate(categories)}
        code_to_attr[attr] = dict(enumerate(categories))
        df[attr] = df[attr].cat.codes

    return code_to_attr, attr_to_code


def map_dataframe(map_dict, df):
    """Map catogorey to its corresponding code, vice versa.

    Args:
        map_dict (dict): The dictionary of the catogorey and code.
        df (pd.DataFrame): The dataframe.

    Returns:
        df (pd.DataFrame): Mapped dataframe
    """
    df = df.copy()
    for attr in map_dict.keys():
        if attr in df.columns:
            df[attr] = df[attr].map(map_dict[attr])
    return df


def make_code_crosswalk(
    source_attr: str,
    target_attr: str,
    code_to_attr: dict,
    attr_to_code: dict,
) -> dict:
    """
    Map encoded states of one variable to encoded states of another
    variable representing the same semantic categories.
    """
    mapping = {}

    for source_code, semantic_value in code_to_attr[source_attr].items():

        if semantic_value not in attr_to_code[target_attr]:
            raise ValueError(
                f"Semantic value {semantic_value!r} from "
                f"{source_attr!r} does not exist in {target_attr!r}."
            )

        mapping[source_code] = attr_to_code[target_attr][semantic_value]

    return mapping


def build_age_group_by_bin(
    age_bin: list,
    code_to_attr: dict[str, dict[str, int]],
    kid_max_age: int = 5,
    ) -> dict[str, list[int]]:
    """Classify each age interval as a kid, minor, or adult group.

    Age intervals are interpreted as ``(lower, upper]`` and are expected to
    align with the kid and adult group boundaries.  An interval that crosses
    a boundary raises ``ValueError``.

    Args:
        age_bin: Ordered age-bin boundaries. Consecutive values define the
            intervals to classify.
        kid_max_age: Maximum age included in the ``"kid"`` group.

    Returns:
        A mapping from each interval's zero-based index to its group name.

    Raises:
        ValueError: If ``kid_max_age`` is outside the range 0 through 17 or
            an interval crosses a group boundary.
    """

    if not isinstance(kid_max_age, int) or not 0 <= kid_max_age < 18:
        raise ValueError("kid_max_age must be an integer from 0 to 17.")

    group_by_bin = {}

    for bin_id, (lower, upper) in enumerate(
        zip(age_bin, age_bin[1:])
    ):
        if upper <= kid_max_age:
            group = "kid"

        elif lower >= GlobleVariables.ADULT_MIN_AGE - 1:
            group = "adult"

        elif lower >= kid_max_age and upper < GlobleVariables.ADULT_MIN_AGE:
            group = "minor"

        else:
            raise ValueError(
                f"Age interval ({lower}, {upper}] "
                "crosses a group boundary."
            )
        group_by_bin[bin_id] = group
    age_states_by_group = {
        "kid": [],
        "minor": [],
        "adult": [],
    }

    for state, bin_id in code_to_attr[GlobleVariables.AGE_CATE].items():
        group = group_by_bin[bin_id]
        age_states_by_group[group].append(state)


    return age_states_by_group