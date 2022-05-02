from typing import Tuple
import numpy as np
import pandas as pd
from textblob import TextBlob


def sentiment_score(text: str) -> Tuple[float, float]:
    if not text or not isinstance(text, str):
        return np.nan, np.nan
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


def add_sentiment(df: pd.DataFrame, based_on: str) -> pd.DataFrame:
    polarity, subjectivity = zip(*[sentiment_score(text) for text in df[based_on]])
    return df.assign(sentiment_polarity=polarity, sentiment_subjectivity=subjectivity)
