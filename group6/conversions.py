import numpy as np
import pandas as pd

def make_tabular(out, ideas):

    idx2label = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'joy',
        4: 'neutral',
        5: 'sadness',
        6: 'surprise'
    }

    df_emo = pd.DataFrame([])
    for i, dist in enumerate(out):
        # extract top label
        scores = [lab['score'] for lab in dist]
        top_label_idx = np.argmax(scores)
        top_label = idx2label[top_label_idx]

        # extract to table
        df_dist = pd.DataFrame(dist)
        df_dist['idea_nr'] = i
        df_dist = df_dist.pivot(index='idea_nr', values='score', columns='label')
        df_dist['top_label'] = top_label
        df_dist['top_label_idx'] = top_label_idx
        # append
        df_emo = pd.concat([df_emo, df_dist])
    
    # merge original df with emotion classifications
    ideas_doc = ideas.reset_index().rename(columns={'index': 'idea_nr'})
    df_idea_emo = ideas_doc.merge(df_emo, on='idea_nr')

    # rename bad cols
    df_idea_emo = df_idea_emo.rename(columns={
        'Status(selectedbyexpert)': 'selected_by_expert',
        'Number.of.Votes': 'n_votes'
    })

    return df_idea_emo
