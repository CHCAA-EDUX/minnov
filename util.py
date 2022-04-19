"""
Utility functions for Lesson 14 of Managing Innocation

author: knielbo
"""
import pandas as pd

def read_data():
    dfs = pd.read_excel('.dat/Lego_subset_22_merge.xlsx', 
        sheet_name=None, 
        engine='openpyxl'
        )
    ideas = dfs["Ideas"].iloc[:,:11]# subset quick fix
    comments = dfs["Comments"].iloc[:,:10]
    ideator = dfs["ideator"]
    
    return (ideas, comments, ideator)

if __name__=="__main__":
    pass