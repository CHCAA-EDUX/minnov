import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_data():
    dfs = pd.read_excel('../.dat/Lego_subset_22_merge.xlsx', 
        sheet_name=None, 
        engine='openpyxl'
        )
    ideas = dfs["Ideas"].iloc[:,:11]# subset quick fix
    comments = dfs["Comments"].iloc[:,:10]
    ideator = dfs["ideator"]
    
    return (ideas, comments, ideator)


def grammar_check(comment, tool):
    matches = tool.check(comment)

    return len(matches)

def min_max_normalization(df, col):
    return (df[col]-df[col].min())/(df[col].max()-df[col].min())


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


def plot_categorical(df, x, y, hue, palette: str = 'Paired', title: str = '',):
    sns.set(rc={'figure.figsize':(14,8)})
    sns.set_style("white")
    plot = sns.barplot(x = df[x], y = df[y], hue = df[hue], palette=palette, dodge=False)
    plot.axes.set_title(title, fontsize = 20)
    plot.set_xlabel('Rank of Idea', fontsize = 14)
    plot.set_ylabel('Score of Idea', fontsize = 14)
    plt.setp(plot.get_legend().get_texts(), fontsize='12')
    plt.setp(plot.get_legend().get_title(), fontsize='14') 
    plot.legend(loc=1)

    for ind, label in enumerate(plot.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.savefig('output/{}.jpg'.format(hue), format='jpeg', dpi=500)
    return plt

def plot_continuous(df, x, y, hue, palette: str = 'Blues', title: str = ''):
    sns.set(rc={'figure.figsize':(14,8)})
    sns.set_style("white")
    plot = sns.barplot(x = df[x], y = df[y], palette=colors_from_values(df[hue], palette ))
    plot.axes.set_title(title, fontsize = 20)
    plot.set_xlabel('Rank of Idea', fontsize = 14)
    plot.set_ylabel('Score of Idea', fontsize = 14)
    for ind, label in enumerate(plot.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.savefig('output/{}.jpg'.format(hue), format='jpeg', dpi=500)
    return plt