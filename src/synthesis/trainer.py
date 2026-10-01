from pgmpy.estimators import HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork
from src.synthesis.containers import ModelSchema, TrainData
import pandas as pd
import logging
import itertools
from typing import Any

logger = logging.getLogger("pgmpy")
logger.setLevel(logging.ERROR)


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
