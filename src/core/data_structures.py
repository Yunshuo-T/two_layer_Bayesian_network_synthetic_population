import pandas as pd
import numpy as np
from dataclasses import dataclass,field
import logging
logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class CalibrationMatrix:
    """Holds compiled design matrices and marginal targets for calibration."""
    x_h: np.ndarray             # Household attributes (s, J_h)
    x_p: np.ndarray             # Individual attributes (s, J_p)
    h_m_ar: np.ndarray          # Household target marginals (J_h,)
    p_m_ar: np.ndarray          # Individual target marginals (J_p,)
    w_initial: np.ndarray       # Initial weights (s,)
    h_sizes: np.ndarray         # Member count per household (s,)
    total_households: float     
    total_individuals: float    

    @property
    def X(self) -> np.ndarray:
        return np.concatenate((self.x_h, self.x_p), axis=1)

    @property
    def t(self) -> np.ndarray:
        return np.concatenate((self.h_m_ar, self.p_m_ar), axis=0)


@dataclass
class CalibrationResult:
    weights: np.ndarray
    is_converged: bool
    iterations: int
    final_error: float


@dataclass
class CalibrationInput:
    h_df: pd.DataFrame
    p_df: pd.DataFrame
    h_marginals: dict[str, dict]
    p_marginals: dict[str, dict]
    h_attrs: list[str]
    p_attrs: list[str]
    h_id_col: str = "H_ID"

class CalibrationMatrixBuilder:
    @classmethod
    def build(
        cls,
        inputs: CalibrationInput,
        joint_attributes: list | None = None
    ) -> CalibrationMatrix:
        """Convert the sample data and marginal data into n-dimension array. 
        Concatenate household and individual array to one single array.For two-layer 
        array based fitting.
        """
        h_marginal_attributes = inputs.h_marginals
        p_marginal_attributes = inputs.p_marginals        
        h_df,h_cols = cls._sample_convert(inputs.h_df,h_marginal_attributes,inputs.h_id_col,joint_attributes)
        p_df,p_cols = cls._sample_convert(inputs.p_df,p_marginal_attributes,inputs.h_id_col,joint_attributes)
        h_sizes = np.array(inputs.p_df.groupby(inputs.h_id_col).size().to_list())
        comb_df = h_df.join(p_df, how='outer').fillna(0)

        zero_cells = comb_df.sum(axis=0)[comb_df.sum(axis=0)==0]
        if not zero_cells.empty:
            zero_cells = zero_cells.index.to_list()
            logger.info(f"The categories {zero_cells} are absent from sample. Dropping them from sample and marginals.")
            h_df = h_df.drop(columns=[col for col in zero_cells if col in h_df.columns])
            p_df = p_df.drop(columns=[col for col in zero_cells if col in p_df.columns])
            
            h_cols = h_df.columns.to_list()
            p_cols = p_df.columns.to_list()
        h_m_ar = cls._marginal_to_array(inputs.h_marginals,h_cols)
        p_m_ar = cls._marginal_to_array(inputs.p_marginals,p_cols)
        x_h = h_df.to_numpy()
        x_p = p_df.to_numpy()
        total_households = sum(inputs.h_marginals[inputs.h_attrs[0]].values())
        total_individuals = sum(inputs.p_marginals[inputs.p_attrs[0]].values())
        w_initial = np.ones(len(h_df)) * (total_households/len(h_df))
        return CalibrationMatrix(
            x_h,            
            x_p,             
            h_m_ar,          
            p_m_ar,         
            w_initial,       
            h_sizes,         
            total_households,     
            total_individuals,
        )
        

    @staticmethod
    def _joint_category(df:pd.DataFrame,joint_attributes:list):
        """

        Args:
            df (pd.DataFrame): The dataframe contains household or individual data in a format that the rows are instances,
            columns are attribtes, values are categories. It must not have NaN value. 
            joint_attributes (list): [[attribute_A,attribute_B]]. The socio-demagraphic attributes such as gender, income etc.
            The order of the attributes must be consistent.

        """
        for joint_attr in joint_attributes:
            df['_'.join(joint_attr)] = ['_'.join(value) for value in df[joint_attr].values.astype(str)]
        return df
    
    @classmethod
    def _sample_convert(cls,df:pd.DataFrame,marginal_attributes:dict,H_index:str,joint_attributes: list | None = None) -> tuple[pd.DataFrame, list]:
        """Conver microdata samples to a tabular format that the columns are the categories 
        of each attributes, and rows of the count of each category within one household.
        
        """
        if joint_attributes:
            df = cls._joint_category(df,joint_attributes)
        df_transformed = df.set_index(H_index)[marginal_attributes.keys()].copy()
        for attr in marginal_attributes:
            df_transformed[attr] = pd.Categorical(df_transformed[attr],marginal_attributes[attr])
        df_transformed = pd.get_dummies(df_transformed)
        col_name = df_transformed.columns.to_list()
        df_transformed = df_transformed.groupby(level=0).sum()

        return df_transformed,col_name
    
    @staticmethod
    def _marginal_to_array(marginals:dict,col_name:list) -> np.ndarray:
        array = []
        for col in col_name:
            key,subkey = col.rsplit('_',1)
            value = marginals[key].get(subkey)
            if value:
                array.append(value)
            else:
                subkey = int(subkey)
                value = marginals[key].get(subkey,10e-6)
                array.append(value)
        return np.array(array)


@dataclass(frozen=True)
class ModelSchema:
    """
    Names of columns used by the two-layer Bayesian-network.
    Training data, fitted models, household samples, and person samples must use
    these names consistently.
    """
    code_to_attr: dict = field(default_factory=dict)
    attr_to_code: dict = field(default_factory=dict)
    H_type: str = 'H_type'
    Head_age: str = 'Head_age'
    Head_gender: str = 'Head_gender'
    Member_rank: str = 'Member_rank'
    p_age: str = 'age_cate'
    p_gender: str = 'SEXE'
    H_id: str = 'H_ID'
    adult_number: str = 'adults'
    minor_number: str = 'minors'
    kid_number: str = 'kids'
    
    @property
    def elementary_attributes(self) -> dict:
        
        return{
            'H':{
            self.H_type,
            self.Head_gender,
            self.Head_age,
            self.adult_number,
            self.minor_number,
            self.kid_number
            },
            'P': {
            self.H_type,
            self.Head_gender,
            self.Head_age,
            self.Member_rank,
            self.p_age,
            self.p_gender
        }
        }
        