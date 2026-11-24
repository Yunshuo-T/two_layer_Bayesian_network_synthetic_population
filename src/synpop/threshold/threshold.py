from synpop import io
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor
import pandas as pd
from collections.abc import Mapping
from typing import Any
from synpop.schema import GlobleVariables
from synpop.training.elementary_attributes import _add_age_category


def make_probability_lookup(
    distribution: pd.DataFrame,
    probability_col: str = "probability",
) -> tuple[dict[tuple[Any, ...], float], list[str]]:
    """Build a reusable household-person probability lookup.
    All columns except probability_col define the individual type. Missing
    combinations return 0.0; missing required attributes raise KeyError.
    """
    attributes = [col for col in distribution.columns if col != probability_col]
    if distribution.duplicated(subset=attributes).any():
        raise ValueError("The probability table contains duplicate individual types.")
    probabilities = dict(
        zip(
            distribution[attributes].itertuples(index=False, name=None),
            distribution[probability_col],
        )
    )
    return probabilities, attributes


def lookup(
    probabilities, attributes, h_dict: Mapping[str, Any], candidate: Mapping[str, Any]
) -> float:
    values = {**h_dict, **candidate}
    key = tuple(values[attr] for attr in attributes)
    return float(probabilities.get(key, 0.0))


def _factor_to_dataframe(
    factor: DiscreteFactor, value_col: str = "probability"
) -> pd.DataFrame:
    """Convert a pgmpy DiscreteFactor into a pandas DataFrame."""
    states = [factor.state_names[var] for var in factor.variables]
    df = pd.MultiIndex.from_product(states, names=factor.variables).to_frame(
        index=False
    )
    df[value_col] = factor.values.ravel()  # type: ignore

    return df


def export_model_joint_distribution(
    h_model: DiscreteBayesianNetwork, p_model: DiscreteBayesianNetwork
) -> pd.DataFrame:
    ele_attributes = (
        GlobleVariables.HEAD_AGE,
        GlobleVariables.HEAD_GENDER,
        GlobleVariables.H_TYPE,
    )

    h_exclude = (
        GlobleVariables.ADULTS,
        GlobleVariables.MINORS,
        GlobleVariables.KIDS,
    )
    p_exclude = (GlobleVariables.MEMBER_RANK,)

    h_keep = [node for node in h_model.nodes if node not in set(h_exclude)]
    p_keep = [node for node in p_model.nodes if node not in set(p_exclude)]

    h_factor = VariableElimination(h_model).query(variables=h_keep, joint=True)
    p_factor = VariableElimination(p_model).query(variables=p_keep, joint=True)

    joint_factor = h_factor * p_factor  # type: ignore
    joint_factor.marginalize(variables=list(ele_attributes), inplace=True)  # type: ignore
    joint_factor.normalize(inplace=True)  # type: ignore

    return _factor_to_dataframe(joint_factor)  # type: ignore


def export_dataframe_joint_distribution(
    h_df: pd.DataFrame, p_df: pd.DataFrame, joint_attributes: list, age_col: str
) -> pd.DataFrame:
    """
    A helper function to export joint distribution of household and individual dataset.

    Args:
        h_df (pd.DataFrame): Household dataframe
        p_df (pd.DataFrame): Individual dataframe
        joint_attributes (list): The attributes that need to be joint from the two datasets
        age_col (str): the column name of the individual age.

    Returns:
        pd.DataFrame: A dataframe containing the joint distribution
    """
    joined_df = pd.merge(p_df, h_df, how="outer", on="H_ID")
    if not (GlobleVariables.AGE_CATE in p_df.columns):
        joined_df = _add_age_category(joined_df, age_col)
        joint_attributes.append(GlobleVariables.AGE_CATE)
        joint_attributes.remove(age_col)
    print(joint_attributes)
    return (
        joined_df[joint_attributes]
        .value_counts(normalize=True)
        .reset_index(name="probability")
    )

def maximum_threshold(df:pd.DataFrame) -> float:
    """
    Calculate the maximum threshold for model pruning.

    Args:
        df (pd.DataFrame): The dataframe of population distribution derived from two-layer
        Bayesian model.
    """
    minimum_prob = 1
    columns = df.columns.to_list()
    columns.remove('probability')
    for col in columns:
        max_prob_per_category = df.groupby([col])['probability'].max()
        prob = max_prob_per_category.min()
        if prob<minimum_prob:
            minimum_prob = prob
    return minimum_prob


if __name__ == "__main__":
    bundle = io.load_bn_bundle(r"output\bn_bundle.pkl")
    df = export_model_joint_distribution(
        h_model=bundle["h_model"], p_model=bundle["p_model"]
    )
    df = df[df["probability"] > 0]
    df.to_csv(r"src/synpop/threshold/individual_distribution/model_distribution.csv")
