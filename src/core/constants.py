class ColumnNames:
    """All column names used by the synthetic population generation."""
    
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
