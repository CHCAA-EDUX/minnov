'''
Functions for the analysis
'''

import pandas as pd
from typing import List
import spacy
from collections import Counter
nlp = spacy.load('en_core_web_sm')

def read_data():
    dfs = pd.read_excel('../.dat/Lego_subset_22_merge.xlsx', 
        sheet_name=None, 
        engine='openpyxl'
        )
    ideas = dfs["Ideas"].iloc[:,:11]# subset quick fix
    comments = dfs["Comments"].iloc[:,:10]
    ideator = dfs["ideator"]
    
    return (ideas, comments, ideator)


def wordcounter(text: str):
    doc = nlp(text)
    return Counter([token.lemma_ for token in doc])


def wordcounter_wcomments(idea: str, comments: List[str]):
    comments.append(idea)
    all_text = ' '.join(comments)
    return wordcounter(all_text)


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx+1))
        print([(vectorizer.get_feature_names_out()[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])