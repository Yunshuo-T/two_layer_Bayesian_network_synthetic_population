from dataclasses import dataclass
import pandas as pd
from pgmpy.estimators import ExpertKnowledge
import itertools
from src.preprocessing import elementary_attributes
from src.core.constants import ColumnNames
from src.core.data_structures import ModelSchema
from src.utils import encoding


@dataclass
class TrainData:
    """Holds the prepared dataset, target nodes, 
    and structural constraints for household or individual model."""
    data:pd.DataFrame
    nodes:list[str]
    expert_knowledge: ExpertKnowledge | None = None


@dataclass
class TrainingBundle:
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
        age_col: str = "AGE",
        gender_col: str = "SEXE",
        h_id_col: str = "H_ID",
    ):  
        processed_h_df,processed_p_df = elementary_attributes.add_elementary_attributes(raw_h_df,raw_p_df,age_col,gender_col,h_id_col)
        cate_code_cols = [col for col in ColumnNames.CATE_CODE_COLS] + [ gender_col ]
        code_to_attr, attr_to_code = encoding.cate_codes(processed_p_df,cate_code_cols)    
        p_mapped = encoding.map_dataframe(attr_to_code,processed_p_df).astype('category')
        h_mapped = encoding.map_dataframe(attr_to_code,processed_h_df).astype('category')
        config = ModelSchema(code_to_attr,attr_to_code)
        h_model_nodes = list(
            dict.fromkeys(h_nodes + list(config.elementary_attributes["H"]))
        )

        p_model_nodes = list(
            dict.fromkeys(p_nodes + list(config.elementary_attributes["P"]))
        )
        forbidden_person_to_household = list(itertools.product(p_nodes, h_nodes))
        forbidden_rank_edges = list(itertools.product(["Member_rank"], h_nodes))
        person_expert_knowledge = ExpertKnowledge(
            forbidden_edges = forbidden_person_to_household + forbidden_rank_edges
        )
        h_spec = TrainData(
            data=h_mapped[h_model_nodes],
            nodes=h_model_nodes,
            expert_knowledge=None,
        )
        p_spec = TrainData(
            data=p_mapped[p_model_nodes],
            nodes=p_model_nodes,
            expert_knowledge=person_expert_knowledge,
        )
        
        return cls(
            h_spec,
            p_spec,
            config
        )