"""
The structure and naming conventions of the model
"""

from dataclasses import dataclass


class SchemaConstants:
    HEAD_COLS = ("Head_age", "Head_gender")
    ADULT_MIN_AGE = 18
    AGE_CATE = "age_cate"
    MEMBER_RANK = "Member_rank"
    HEAD_AGE = "Head_age"
    HEAD_GENDER = "Head_gender"
    ADULTS = "adults"
    MINORS = "minors"
    KIDS = "kids"
    H_TYPE = "H_type"

    HOUSEHOLD_COLS = frozenset({H_TYPE, HEAD_AGE, HEAD_GENDER, ADULTS, MINORS, KIDS})
    PERSON_COLS = frozenset({MEMBER_RANK, AGE_CATE, H_TYPE, HEAD_AGE, HEAD_GENDER})
    CATE_CODE_COLS = [HEAD_AGE, HEAD_GENDER, H_TYPE, MEMBER_RANK, AGE_CATE]


@dataclass(frozen=True)
class ModelSchema:
    """
    This schema stores only names that may differ between source datasets and
    the category mappings learned during preprocessing.
    """

    code_to_attr: dict
    attr_to_code: dict
    gender_col: str
    h_id_col: str
    age_states_by_group: dict

    @property
    def elementary_attributes(self) -> dict[str, set[str]]:
        return {
            "H": {
                SchemaConstants.H_TYPE,
                SchemaConstants.HEAD_AGE,
                SchemaConstants.HEAD_GENDER,
                SchemaConstants.ADULTS,
                SchemaConstants.MINORS,
                SchemaConstants.KIDS,
            },
            "P": {
                SchemaConstants.H_TYPE,
                SchemaConstants.HEAD_AGE,
                SchemaConstants.HEAD_GENDER,
                SchemaConstants.MEMBER_RANK,
                SchemaConstants.AGE_CATE,
                self.gender_col,
            },
        }
