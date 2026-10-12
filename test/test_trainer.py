from synpop.training import trainer
from synpop.schema import ColumnNames
import pytest
import itertools
import pandas as pd
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
    
@pytest.fixture
def data_frame():
    return {
        'household': pd.DataFrame(
            {
            "household_id": [1, 2],
            "household_type": ["single", "family"],
            }
        ),
        'person': pd.DataFrame(
            {
            "household_id": [1, 2],
            "gender": ["F", "M"],
            }
        )
    }
    
@pytest.fixture
def mock_trainer():
    mock_config = Mock()
    return trainer.Trainer(config=mock_config)
    
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "A": [0, 0, 0, 1, 1, 1, 0, 1],
        "B": [0, 0, 1, 1, 1, 0, 0, 1],
    })

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
    
    def test_construct_config(self, data_frame,monkeypatch):
        data_frame = data_frame
        h_df = data_frame['household']
        p_df = data_frame['person']
        
        # Expected encoding dictionaries
        code_to_attr = {"gender": {0: "F", 1: "M"}}
        attr_to_code = {"gender": {"F": 0, "M": 1}}

        monkeypatch.setattr(
            trainer.encoding,
            "cate_codes",
            lambda *args: (code_to_attr, attr_to_code),
        )
        monkeypatch.setattr(
            trainer.encoding,
            "map_dataframe",
            lambda mapping, df: df.copy(),
        )
        monkeypatch.setattr(
            trainer,
            "ModelSchema",
            lambda *args: args,
        )
        p_mapped, h_mapped, config = trainer._construct_config(
        h_df, p_df, "gender", "household_id"
        )

        pd.testing.assert_frame_equal(p_mapped, p_df.astype("category"))
        pd.testing.assert_frame_equal(h_mapped, h_df.astype("category"))
        assert config == (
            code_to_attr, attr_to_code, "gender", "household_id"
        )



class TestTrainer:
    
    def test_validate_dataframe(self,sample_data,mock_trainer):
        df_missing = sample_data[["B"]]
        with pytest.raises(ValueError) as exc_info:
            mock_trainer._validate_dataframe(df_missing,{"A", "B"})
        
        assert "Missing required columns: ['A']" in str(exc_info.value)
        
        df_empty = pd.DataFrame({'A':[],'B':[]})
        with pytest.raises(ValueError) as exc_info:
            mock_trainer._validate_dataframe(df_empty,{"A", "B"})
        
        assert "Input data must contain at least one row." in str(exc_info.value)
        
        df_null = sample_data.copy()
        df_null.loc[df_null.index[0], "A"] = None 
        with pytest.raises(ValueError) as exc_info:
            mock_trainer._validate_dataframe(df_null,{"A", "B"})
        
        assert "Missing values are not supported in required columns: ['A']" in str(exc_info.value)