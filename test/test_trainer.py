from synpop.training import trainer
from synpop.schema import ColumnNames
import pytest
import itertools
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import Mock

H_ID = "household_id"
AGE = "age"
GENDER = "gender"


@pytest.fixture
def elementary_attrs():
    return {
        "H": {"household_type", "household_size"},
        "P": {ColumnNames.MEMBER_RANK, ColumnNames.AGE_CATE},
    }

@pytest.fixture
def p_nodes():
    return ["education", "occupation"]


@pytest.fixture
def default_expert_knowledge():
    return {
        "household": {
            "forbidden_edges": None,
            "required_edges": None,
        },
        "person": {
            "forbidden_edges": None,
            "required_edges": None,
        },
    }
    

class TestBundleConstruction:
    
    def test__build_expert_knowledge_default(
        self,
        p_nodes,
        elementary_attrs,
        default_expert_knowledge
    ):
        household_ek,person_ek = trainer._build_expert_knowledge(
        p_nodes=p_nodes,
        elementary_attributes=elementary_attrs,
        user_expert_knowledge=default_expert_knowledge,
        )
        expected_person_forbidden = set(itertools.product(
        [*p_nodes, ColumnNames.MEMBER_RANK, ColumnNames.AGE_CATE],
        elementary_attrs["H"],
        ))
        
        assert set(person_ek.forbidden_edges) == expected_person_forbidden
        assert set(person_ek.required_edges) == set()

        assert set(household_ek.forbidden_edges) == set()
        assert set(household_ek.required_edges) == set()
        
    def test_resolve_model_nodes(
        self,
        p_nodes,
        elementary_attrs  
    ):
        nodes = trainer._resolve_model_nodes(
            user_nodes=p_nodes,
            elementary_nodes=elementary_attrs["P"]
        )
        
        assert nodes == ["education", "occupation",ColumnNames.MEMBER_RANK, ColumnNames.AGE_CATE]
    
    def test_construct_config(self, monkeypatch):
        h_df = pd.DataFrame({
            "household_id": [1, 2],
            "household_type": ["single", "family"],
        })

        p_df = pd.DataFrame({
            "household_id": [1, 2],
            "gender": ["F", "M"],
        })

        # Expected encoding dictionaries
        code_to_attr = {"gender": {0: "F", 1: "M"}}
        attr_to_code = {"gender": {"F": 0, "M": 1}}

        # Expected mapped DataFrames
        expected_p = p_df.copy()
        expected_p["gender"] = [0, 1]

        expected_h = h_df.copy()

        # Mock dependencies
        mock_cate_codes = Mock(
            return_value=(code_to_attr, attr_to_code)
        )
        mock_map_dataframe = Mock(
            side_effect=[expected_p, expected_h]
        )
        mock_schema = Mock()

        monkeypatch.setattr(
            trainer.encoding, "cate_codes", mock_cate_codes
        )
        monkeypatch.setattr(
            trainer.encoding, "map_dataframe", mock_map_dataframe
        )
        monkeypatch.setattr(
            trainer, "ModelSchema", mock_schema
        )

        # Act
        p_mapped, h_mapped, config = trainer._construct_config(
            processed_h_df=h_df,
            processed_p_df=p_df,
            gender_col="gender",
            h_id_col="household_id",
        )

        # Assert: categorical codes were generated correctly
        mock_cate_codes.assert_called_once_with(
            p_df,
            [*ColumnNames.CATE_CODE_COLS, "gender"],
        )

        # Assert: encoding was applied in the correct order
        assert mock_map_dataframe.call_count == 2

        calls = mock_map_dataframe.call_args_list

        assert calls[0].args[0] is attr_to_code
        assert calls[0].args[1] is p_df

        assert calls[1].args[0] is attr_to_code
        assert calls[1].args[1] is h_df

        # Assert: output DataFrames are correct
        assert_frame_equal(
            p_mapped, expected_p.astype("category")
        )
        assert_frame_equal(
            h_mapped, expected_h.astype("category")
        )

        # Assert: ModelSchema received the expected arguments
        mock_schema.assert_called_once_with(
            code_to_attr,
            attr_to_code,
            "gender",
            "household_id",
        )

        assert config is mock_schema.return_value
            
            