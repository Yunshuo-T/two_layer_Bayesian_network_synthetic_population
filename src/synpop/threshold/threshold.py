from scipy.stats import poisson
THRESHOLD = 10e-6
from synpop import io
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor
import itertools
import pandas as pd
from pathlib import Path
from synpop.schema import ColumnNames
from synpop.training.elementary_attributes import _add_age_category


def _factor_to_dataframe(factor:DiscreteFactor, value_col:str ="probability") -> pd.DataFrame:
    """Convert a pgmpy DiscreteFactor into a pandas DataFrame."""
    states = [factor.state_names[var] for var in factor.variables]
    df = pd.MultiIndex.from_product(states, names=factor.variables).to_frame(
        index=False
    )
    df[value_col] = factor.values.ravel() # type: ignore
    
    return df


def two_layer_joint_distribution(h_model:DiscreteBayesianNetwork,p_model:DiscreteBayesianNetwork) -> pd.DataFrame:
    ele_attributes = (ColumnNames.HEAD_AGE,ColumnNames.HEAD_GENDER,ColumnNames.H_TYPE,)
    
    h_exclude = (ColumnNames.ADULTS,ColumnNames.MINORS,ColumnNames.KIDS,)
    p_exclude = (ColumnNames.MEMBER_RANK,)
    
    h_keep = [node for node in h_model.nodes if node not in set(h_exclude)]
    p_keep = [node for node in p_model.nodes if node not in set(p_exclude)]
    
    h_factor = VariableElimination(h_model).query(variables=h_keep,joint=True)
    p_factor = VariableElimination(p_model).query(variables=p_keep,joint=True)
    
    joint_factor = h_factor * p_factor # type: ignore
    joint_factor.marginalize(variables=list(ele_attributes), inplace=True)# type: ignore
    joint_factor.normalize(inplace=True)# type: ignore

    return _factor_to_dataframe(joint_factor)# type: ignore

def export_joint_distribution(h_df:pd.DataFrame,p_df:pd.DataFrame,joint_attributes:list,age_col:str)-> pd.DataFrame:
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
    joined_df = pd.merge(
        p_df,
        h_df,
        how='outer',
        on='H_ID'
    )
    if not (ColumnNames.AGE_CATE in p_df.columns):
        joined_df = _add_age_category(joined_df,age_col)
    joint_attributes.append(ColumnNames.AGE_CATE)
    joint_attributes.remove(age_col)
    return joined_df[joint_attributes].value_counts(normalize=True).reset_index(name='probability')
    

if __name__ == "__main__":
    bundle = io.load_bn_bundle(r'output\bn_bundle.pkl')
    df = two_layer_joint_distribution(h_model=bundle['h_model'],p_model=bundle['p_model'])
    df.to_csv(r"output/refacotr_bn_joint.csv")