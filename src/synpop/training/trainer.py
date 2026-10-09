import logging
import itertools
from typing import Any
import pandas as pd
from dataclasses import dataclass
from pgmpy.estimators import ExpertKnowledge, HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork

from synpop.training import elementary_attributes
from synpop.schema import ColumnNames, ModelSchema


from synpop.utils import encoding

logger = logging.getLogger("pgmpy")
logger.setLevel(logging.ERROR)


@dataclass
class TrainData:
    """Holds the prepared dataset, target nodes,
    and structural constraints for household or individual model."""

    data: pd.DataFrame
    nodes: list[str]
    expert_knowledge: ExpertKnowledge | None = None


class Trainer:

    def __init__(self, config: ModelSchema) -> None:
        self.config = config

    def _validate_dataframe(
        self, data: pd.DataFrame, elementary_attributes: set[str]
    ) -> None:
        """Validate that data is non-empty and contains complete required columns."""
        missing = elementary_attributes.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if data.empty:
            raise ValueError("Input data must contain at least one row.")

        null_columns = (
            data[list(elementary_attributes)]
            .columns[data[list(elementary_attributes)].isna().any()]
            .tolist()
        )
        if null_columns:
            raise ValueError(
                f"Missing values are not supported in required columns: {null_columns}"
            )

    def _learn_structure(
        self,
        data: pd.DataFrame,
        score: str = "bic-d",
        show_progress: bool = False,
        expert_knowledge: Any | None = None,
    ):
        """
        Learn a directed acyclic graph from discrete training data.

        Args:
            data: Discrete observations, with one model variable per column.
            score: pgmpy score name, such as ``"bic-d"``, ``"aic-d"``, or
            ``"bdeu"``.
            expert_knowledge: Optional pgmpy ``ExpertKnowledge`` constraints.
            show_progress: Whether pgmpy should show estimation progress.

        Returns:
            A learned pgmpy DAG.
        """
        return HillClimbSearch(data=data).estimate(
            scoring_method=score,
            expert_knowledge=expert_knowledge,
            show_progress=show_progress,
        )

    def _learn_parameters(self, dag, data: pd.DataFrame, dag_path: str | None = None):  # type: ignore
        """
        Fit conditional probability distributions using maximum likelihood estimation.

        Args:
            dag: Learned DAG that defines the network structure.
            data: Training observations containing every DAG node.
            dag_path: Optional path prefix for an EPS Graphviz export of DAG.

        Returns:
            A fitted and validated ``DiscreteBayesianNetwork``.
        """
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(dag.nodes())
        model.add_edges_from(dag.edges())
        model.fit(data)  # Maximum Likelihood Estimator (default)

        if dag_path:
            model_graphviz = model.to_graphviz()
            model_graphviz.draw(f"{dag_path}.eps", prog="dot")
        return model

    def train_model(
        self,
        train_data: TrainData,
        type: str,
        dag_path: str | None = None,
        score="aic-d",
        show_progress: bool = False,
    ):
        """
        Learn a Bayesian-network structure and fit parameters using maximum likelihood.

        Required and forbidden edges should be supplied through
        ``expert_knowledge``.

        Args:
            data: Discrete training data.
            type: Household model or person model. "H" or "P"
            expert_knowledge: Optional pgmpy structural constraints.
            required_columns: list[str],  A list of attribute name used for model training.
            dag_path: Optional EPS graph output path prefix.
            score: Structure-learning score.
            show_progress: Whether to show hill-climbing progress.
        Returns:
            A fitted ``DiscreteBayesianNetwork``.
        """
        self._validate_dataframe(
            train_data.data, self.config.elementary_attributes[type]
        )
        dag = self._learn_structure(
            train_data.data,
            score=score,
            expert_knowledge=train_data.expert_knowledge,
            show_progress=show_progress,
        )
        return self._learn_parameters(dag, train_data.data, dag_path=dag_path)

    def export_cpd(
        self,
        attribute: str,
        model: DiscreteBayesianNetwork,
        path: str,
    ) -> pd.DataFrame:
        """
        Export one conditional probability distribution to CSV.

        Args:
            attribute: Variable name whose CPD should be exported.
            model: Fitted Bayesian network.
            path: Output path prefix; ``.csv`` is appended.

        Returns:
            A data frame with child states as rows and parent-state combinations
            as columns.
        """

        cpd = model.get_cpds(attribute)
        if cpd is None:
            raise ValueError(f"Model has no CPD for '{attribute}'.")

        child_var = cpd.variable  # type: ignore
        parents = list(cpd.variables[1:])  # type: ignore
        child_states = cpd.state_names[child_var]  # type: ignore
        parent_states = [cpd.state_names[p] for p in parents]  # type: ignore
        if parents:
            col_index = pd.MultiIndex.from_tuples(
                list(itertools.product(*parent_states)), names=parents  # type: ignore
            )
        else:
            col_index = ["Prob"]

        prob_matrix = cpd.values.reshape(len(child_states), -1)  # type: ignore

        df_cpd = pd.DataFrame(prob_matrix, index=child_states, columns=col_index)
        df_cpd.index.name = child_var
        df_cpd.to_csv(f"{path}.csv")

        return df_cpd


def _construct_config(
    processed_h_df: pd.DataFrame,
    processed_p_df: pd.DataFrame,
    gender_col,
    h_id_col,
):
    cate_code_cols = [col for col in ColumnNames.CATE_CODE_COLS] + [gender_col]
    code_to_attr, attr_to_code = encoding.cate_codes(processed_p_df, cate_code_cols)
    p_mapped = encoding.map_dataframe(attr_to_code, processed_p_df).astype("category")
    h_mapped = encoding.map_dataframe(attr_to_code, processed_h_df).astype("category")
    config = ModelSchema(code_to_attr, attr_to_code, gender_col, h_id_col)
    return p_mapped, h_mapped, config


def _resolve_model_nodes(user_nodes: list[str], elementary_nodes: set[str]) -> list[str]:
    return list(dict.fromkeys([*user_nodes, *sorted(elementary_nodes)]))


def _build_expert_knowledge(
    p_nodes: list[str],
    elementary_attributes: dict[str, set[str]],
    user_expert_knowledge: dict,
):
    forbidden_edges = itertools.product(
        [*p_nodes, ColumnNames.MEMBER_RANK, ColumnNames.AGE_CATE],
        elementary_attributes["H"],
    )
    person_config = user_expert_knowledge["person"]
    household_config = user_expert_knowledge["household"]
    
    person_expert_knowledge = ExpertKnowledge(
        forbidden_edges=list(forbidden_edges)
        + (person_config["forbidden_edges"] or []),
        required_edges=person_config["required_edges"] or [],
    )
    household_expert_knowledge = ExpertKnowledge(
        forbidden_edges=household_config["forbidden_edges"] or [],
        required_edges=household_config["required_edges"] or [],
    )

    return household_expert_knowledge, person_expert_knowledge


@dataclass
class TrainingBundle:
    """A bundle for training data and training configuration"""

    household: TrainData
    person: TrainData
    config: ModelSchema

    @classmethod
    def construct_bundle(
        cls,
        raw_h_df: pd.DataFrame,
        raw_p_df: pd.DataFrame,
        h_nodes: list[str],
        p_nodes: list[str],
        age_col: str,
        gender_col: str,
        h_id_col: str,
        expert_knowledge: dict,
        age_bin: list | None = None,
        composition_cols: dict[str, str] | None = None,
    ):
        processed_h_df, processed_p_df = (
            elementary_attributes.add_elementary_attributes(
                raw_h_df=raw_h_df,
                raw_p_df=raw_p_df,
                age_col=age_col,
                gender_col=gender_col,
                h_id_col=h_id_col,
                age_bin=age_bin,
                composition_cols=composition_cols,
            )
        )
        p_mapped, h_mapped, config = _construct_config(
            processed_h_df=processed_h_df,
            processed_p_df=processed_p_df,
            gender_col=gender_col,
            h_id_col=h_id_col,
        )
        h_model_nodes = _resolve_model_nodes(h_nodes, config.elementary_attributes["H"])
        p_model_nodes = _resolve_model_nodes(p_nodes, config.elementary_attributes["P"])

        household_expert_knowledge, person_expert_knowledge = _build_expert_knowledge(
            p_nodes=p_nodes,
            elementary_attributes=config.elementary_attributes,
            user_expert_knowledge=expert_knowledge,
        )

        h_spec = TrainData(
            data=h_mapped[h_model_nodes],
            nodes=h_model_nodes,
            expert_knowledge=household_expert_knowledge,
        )
        p_spec = TrainData(
            data=p_mapped[p_model_nodes],
            nodes=p_model_nodes,
            expert_knowledge=person_expert_knowledge,
        )

        return cls(h_spec, p_spec, config)
