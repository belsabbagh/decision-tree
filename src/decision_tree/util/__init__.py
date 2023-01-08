""" Utility functions for the dataframes. """
import pandas as pd


def exclude_df_col(df: pd.DataFrame, f: str) -> pd.DataFrame:
    """Excludes a feature from a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to exclude the feature from.
        f (str): The feature to exclude.

    Returns:
        pd.DataFrame: The dataframe without the feature.
    """
    return df.loc[:, ~df.columns.isin([f])]
