import pandas as pd
import string
from nltk.stem.wordnet import WordNetLemmatizer

def read_data():
    dfs = pd.read_excel('../.dat/Lego_subset_22_merge.xlsx', 
        sheet_name=None, 
        engine='openpyxl'
        )
    ideas = dfs["Ideas"].iloc[:,:11]# subset quick fix
    comments = dfs["Comments"].iloc[:,:10]
    ideator = dfs["ideator"]
    
    return (ideas, comments, ideator)

def clean_text(comment, stop_words):
    words = [text.split() for text in comment]
    table = str.maketrans('', '', string.punctuation)
    stripped = [[w.translate(table) for w in word] for word in words]
    comment = [[x.lower() for x in comment] for comment in stripped]
    comment = [[x for x in list if x not in stop_words] for list in comment]
    lem = WordNetLemmatizer()
    lemmatized = [[lem.lemmatize(word) for word in list] for list in comment]
    done = [[' '.join(i)] for i in lemmatized]
    
    return done
