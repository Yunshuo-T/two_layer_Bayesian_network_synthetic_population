"""
The structures of input and output data for the pipeline.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationMatrix:
    """Holds compiled design matrices and marginal targets for calibration."""

    x_h: np.ndarray  # Household attributes (s, J_h)
    x_p: np.ndarray  # Individual attributes (s, J_p)
    h_m_ar: np.ndarray  # Household target marginals (J_h,)
    p_m_ar: np.ndarray  # Individual target marginals (J_p,)
    w_initial: np.ndarray  # Initial weights (s,)
    h_sizes: np.ndarray  # Member count per household (s,)
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
        joint_attributes: list | None = None,
        weight_col: str | None = None,
    ) -> CalibrationMatrix:
        """Convert the sample data and marginal data into n-dimension array.
        Concatenate household and individual array to one single array.For two-layer
        array based fitting.
        """
        h_marginal_attributes = inputs.h_marginals
        p_marginal_attributes = inputs.p_marginals
        total_households = cls._marginal_consistency(inputs.h_marginals, inputs.h_attrs)
        total_individuals = cls._marginal_consistency(
            inputs.p_marginals, inputs.p_attrs
        )

        h_df, h_cols = cls._sample_convert(
            inputs.h_df, h_marginal_attributes, inputs.h_id_col, joint_attributes
        )
        p_df, p_cols = cls._sample_convert(
            inputs.p_df, p_marginal_attributes, inputs.h_id_col, joint_attributes
        )

        if not h_df.index.equals(p_df.index):
            raise ValueError(
                "Household and individual datasets must have the same household IDs"
            )

        h_m_ar = cls._marginal_to_array(inputs.h_marginals, h_cols)
        p_m_ar = cls._marginal_to_array(inputs.p_marginals, p_cols)
        x_h = h_df.to_numpy()
        x_p = p_df.to_numpy()
        h_sizes = np.array(inputs.p_df.groupby(inputs.h_id_col).size().to_list())
        if weight_col:
            w_initial = h_df[weight_col].to_numpy()
        else:
            w_initial = np.ones(len(h_df)) * (total_households / len(h_df))

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
    def _joint_category(df: pd.DataFrame, joint_attributes: list):
        """

        Args:
            df (pd.DataFrame): The dataframe contains household or individual data in a format that the rows are records,
            columns are attributes, values are categories. It must not have NaN value.
            joint_attributes (list): [[attribute_A,attribute_B]]. The category of the attributes should not contain underscore. The socio-demagraphic attributes such as gender, income etc.
            The order of the attributes must be consistent.

        """
        for joint_attr in joint_attributes:
            df["_".join(joint_attr)] = [
                "_".join(value) for value in df[joint_attr].values.astype(str)
            ]
        return df

    @classmethod
    def _sample_convert(
        cls,
        df: pd.DataFrame,
        marginal_attributes: dict,
        H_index: str,
        joint_attributes: list | None = None,
    ) -> tuple[pd.DataFrame, list]:
        """Return per-household category counts and the columns present in the sample."""
        if joint_attributes:
            df = cls._joint_category(df, joint_attributes)
        df_transformed = df.set_index(H_index)[marginal_attributes.keys()].copy()
        for attr in marginal_attributes:
            df_transformed[attr] = pd.Categorical(
                df_transformed[attr].astype("string"), marginal_attributes[attr]
            )
        attributes = list(marginal_attributes)
        indicators = pd.concat(
            [pd.get_dummies(df_transformed[attr]) for attr in attributes],
            axis=1,
            keys=attributes,
            names=["attribute", "category"],
        )
        counts = indicators.groupby(level=0).sum()
        counts = cls._drop_empty_columns(counts, marginal_attributes)
        return counts, counts.columns.to_list()

    @staticmethod
    def _marginal_to_array(marginals: dict, columns: list) -> np.ndarray:
        return np.array(
            [marginals[attribute][category] for attribute, category in columns]
        )

    @staticmethod
    def _marginal_consistency(marginals: dict, attrs: list[str]):
        reference = attrs[0]
        total = sum(marginals[reference].values())

        for attr in attrs[1:]:
            attr_total = sum(marginals[attr].values())
            if attr_total != total:
                raise ValueError(
                    f"Marginal {attr!r} has totals {attr_total}, "
                    f"but {reference!r} has totals {total}."
                )
        return total

    @staticmethod
    def _drop_empty_columns(
        counts: pd.DataFrame,
        marginals: dict,
    ) -> pd.DataFrame:
        empty_cols = counts.columns[counts.sum(axis=0).eq(0)].to_list()
        if not empty_cols:
            return counts

        unsupported = [
            (attr, category)
            for attr, category in empty_cols
            if marginals[attr][category] != 0
        ]
        if unsupported:
            logger.warning(
                "Dropping empty columns with non-zero targets: %s",
                unsupported,
            )
        return counts.drop(columns=empty_cols)
