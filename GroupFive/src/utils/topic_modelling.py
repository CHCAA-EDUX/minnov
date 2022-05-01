from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.text import lemmatize_texts


class NMFTopics:
    def __init__(self, n_topics=5, cutoff_frequency: float = 0.2):
        self.tf_idf = TfidfVectorizer(max_df=cutoff_frequency, min_df=1)
        self.nmf = NMF(n_components=n_topics)

    def fit(self, texts: Iterable[str]):
        clean_corpus = [" ".join(text) for text in lemmatize_texts(texts)]
        self.tf_idf.fit(clean_corpus)
        corpus_matrix = self.tf_idf.transform(clean_corpus)
        self.nmf.fit(corpus_matrix)
        return self

    def transform(self, texts: Iterable[str]):
        clean_texts = [" ".join(text) for text in lemmatize_texts(texts)]
        text_matrix = self.tf_idf.transform(clean_texts)
        return self.nmf.transform(text_matrix)

    def get_topics(self, top_words=10):
        topics = []
        feature_names = self.tf_idf.get_feature_names_out()
        components = self.nmf.components_
        for topic_index in range(self.nmf.n_components_):
            topic = components[topic_index]
            top_5_features = np.argsort(-topic)[:top_words]
            topics.append(
                {
                    str(feature_names[feature]): topic[feature]
                    for feature in top_5_features
                }
            )
        return topics


def add_nmf_topics(
    df: pd.DataFrame, based_on: str, topic_model: NMFTopics
) -> pd.DataFrame:
    topic_matrix = topic_model.transform(df[based_on])
    topic_labels = np.argmax(topic_matrix, axis=1)
    return df.assign(**{f"{based_on}_topic": topic_labels})
