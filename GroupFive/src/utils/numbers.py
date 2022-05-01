from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats

from sklearn.decomposition import PCA


def reduce_pca(X: np.ndarray, dim: int = 10, verbose: bool = True):
    """
    Reduces data matrix X to a given number of features using PCA.
    If verbose is set to true it prints the total amount of variance
    explained after dimensionality reduction.
    """
    pca = PCA(dim).fit(X)
    if verbose:
        total_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Total explained variance: {total_variance}")
    return pca.transform(X)


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.nanmax(a)


def normalize_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    mapping = {column: normalize(df[column].to_numpy()) for column in columns}
    return df.assign(**mapping)


def corr_test(df: pd.DataFrame, column1: str, column2: str) -> None:
    print(scipy.stats.spearmanr(df[column1], df[column2], nan_policy="omit"))
    px.scatter(df, x=column1, y=column2, trendline="ols").show()


def np_ify(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_list())
