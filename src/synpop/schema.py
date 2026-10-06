"""
The configurations for the pipeline.
"""

from dataclasses import dataclass


class ColumnNames:
    HEAD_COLS = ("Head_age", "Head_gender")
    COMPOSITION_COLS = ("adults", "minors", "kids")

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

    @property
    def elementary_attributes(self) -> dict[str, set[str]]:
        return {
            "H": {
                ColumnNames.H_TYPE,
                ColumnNames.HEAD_AGE,
                ColumnNames.HEAD_GENDER,
                ColumnNames.ADULTS,
                ColumnNames.MINORS,
                ColumnNames.KIDS,
            },
            "P": {
                ColumnNames.H_TYPE,
                ColumnNames.HEAD_AGE,
                ColumnNames.HEAD_GENDER,
                ColumnNames.MEMBER_RANK,
                ColumnNames.AGE_CATE,
                self.gender_col,
            },
        }
