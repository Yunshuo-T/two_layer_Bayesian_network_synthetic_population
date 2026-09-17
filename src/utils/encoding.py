import pandas as pd


def cate_codes(df:pd.DataFrame, attributes:list, explicit_orders: dict = None): # type: ignore
    df = df.copy() 
    attr_to_code = {}
    code_to_attr = {}
    
    if explicit_orders is None:
        explicit_orders = {}

    for attr in attributes:
        if attr in explicit_orders:
            cat_type = pd.CategoricalDtype(categories=explicit_orders[attr], ordered=True)
            df[attr] = df[attr].astype(cat_type)
        else:
            df[attr] = df[attr].astype('category')
            
        categories = df[attr].cat.categories
        attr_to_code[attr] = {val: i for i, val in enumerate(categories)}
        code_to_attr[attr] = dict(enumerate(categories))
        df[attr] = df[attr].cat.codes
            
    return code_to_attr, attr_to_code

def map_dataframe(map_dict,df):
    """Map catogorey to its corresponding code, vice versa.

    Args:
        map_dict (dict): The dictionary of the catogorey and code.
        df (pd.DataFrame): The dataframe.

    Returns:
        df (pd.DataFrame): Mapped dataframe
    """
    df = df.copy()
    for attr in map_dict.keys():
        if attr in df.columns:
            df[attr] = df[attr].map(map_dict[attr])
    return df


def make_code_crosswalk(
    source_attr: str,
    target_attr: str,
    code_to_attr: dict,
    attr_to_code: dict,
) -> dict:
    """
    Map encoded states of one variable to encoded states of another
    variable representing the same semantic categories.
    """
    mapping = {}

    for source_code, semantic_value in code_to_attr[source_attr].items():

        if semantic_value not in attr_to_code[target_attr]:
            raise ValueError(
                f"Semantic value {semantic_value!r} from "
                f"{source_attr!r} does not exist in {target_attr!r}."
            )

        mapping[source_code] = (
            attr_to_code[target_attr][semantic_value]
        )

    return mapping

