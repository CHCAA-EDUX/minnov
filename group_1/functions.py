'''
Functions used for analysis and visualizations
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pysentimiento as ps
import functools
import os
from typing import Optional, List

################
# Read in data #
################

def read_data():
    dfs = pd.read_excel('../.dat/Lego_subset_22_merge.xlsx', 
        sheet_name=None, 
        engine='openpyxl'
        )
    ideas = dfs["Ideas"].iloc[:,:11]# subset quick fix
    comments = dfs["Comments"].iloc[:,:10]
    ideator = dfs["ideator"]
    
    return (ideas, comments, ideator)

############
# Plotting #
############

def plot_votes_vs_x(df: pd.DataFrame, 
                    x:str, 
                    x_label:str, 
                    y:str = "Number.of.Votes", 
                    y_label:str = "Number of votes", 
                    title:str = "", 
                    x_tick_labels=None, 
                    rotation:int = 0,
                    add_reg_line:bool = False):
    """Function for visualizing the LEGO ideas data. Tailored to a handful specific situations

    Args:
        df (pd.DataFrame): Dataframe with the data
        x (str): variable to plot on x-axis
        x_label (str): label for the x-axis
        y (str, optional): variable to plot on y-axis. Defaults to "Number.of.Votes".
        y_label (str, optional): label for the y-axis. Defaults to "Number of votes".
        title (str, optional): title for the plot. Defaults to ""
        x_labels (list, optional): list of labels to use for x-axis. Defaults to None.
        rotation (int, optional): degree of rotation for x-axis labels. Defaults to 0.

    Returns:
        fig: The figure with the specified properties
    """    
    fig, ax = plt.subplots(figsize=(18,8))
    if add_reg_line:
        ax = sns.regplot(x=x, y=y, data=df, scatter_kws={'s':40})
    else:
        ax = sns.scatterplot(x=x, y=y, data=df, s=70)
    
    # If x axis labels and a rotation is provided
    if x_tick_labels and rotation != 0: 
        ax.set_xticklabels(x_tick_labels, rotation = rotation, ha="right", fontsize=12)
    
    # If x axis labels are provided as strings but no rotation is provided
    elif x_tick_labels and isinstance(x_tick_labels[0], str) and rotation == 0:
        ax.set_xticklabels(x_tick_labels, fontsize=12)

    # If x axis labels are proved as integers 
    elif x_tick_labels and isinstance(x_tick_labels[0], int):
        ax.set_xticks(x_tick_labels)

    ax.set_xlabel(x_label, fontsize= 20)
    ax.set_ylabel(y_label, fontsize= 20)
    ax.set_title(title, fontsize= 20)
    ax.tick_params(axis="x", labelsize=12)
    plt.grid(linestyle="--", linewidth=0.3)
    return fig


#############
# Sentiment #
#############

def english_sentiment(column:pd.Series, model):
    """This function extracts Pysentimentio sentiment.
    Returns a tuple with the predicted label and the associated probability

    Args:
        column (pd.Series): column in dataframe to apply sentiment to
        model (pysentimiento.analyzer.SentimentAnalyzer): Pysentimention analyzer object

    Returns:
        tuple: (polarity label, polarity prob) or ("nan", "nan") for "nan" input
    """    
    if pd.isna(column):
        return (float("nan"), float("nan"))
    output = model.predict(column)
    return (output.output, output.probas[output.output])

def bert_scores_en(df:pd.DataFrame, col:str, out_path:Optional[str]=None):
    # Prepare file

    # Apply using analyzer
    analyzer = ps.SentimentAnalyzer(lang="en")
    partial_func = functools.partial(english_sentiment, model=analyzer)
    sentiment = list(map(partial_func, df[col]))
    sents = list(zip(*sentiment))

    # Dictionary for getting polarity score
    pol_dict = {"POS": 1,
                "NEU": 0,
                "NEG": -1}

    # Add to df
    df["polarity"] = sents[0]
    df["polarity_score"] = [pol_dict[label] if not pd.isna(label) else label for label in df["polarity"]]
    df["polarity_prob"] = sents[1]

    if out_path:
        df.to_csv(f'{out_path}.csv')
    return df