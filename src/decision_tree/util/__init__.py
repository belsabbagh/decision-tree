import pandas as pd


def exclude_df_col(df: pd.DataFrame, f: str) -> pd.DataFrame:
    return df.loc[:, ~df.columns.isin([f])]
