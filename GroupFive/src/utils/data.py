from typing import Dict

import pandas as pd
import pycountry

from utils.text import normalize


Sheet = Dict[str, pd.DataFrame]


def clean_country_name(country: str) -> str:
    """Turns a mess of country codes and names into uppercase country names"""
    country = country.upper()
    country_object = pycountry.countries.get(alpha_2=country, default=None)
    if country_object is not None:
        return country_object.name.upper()
    return country


def load_sheet(path: str) -> Sheet:
    raw_sheet = pd.read_excel(
        path,
        sheet_name=None,  # Load all sheets
    )
    ideas = raw_sheet["Ideas"].rename(
        {
            "Submission.ID": "submission_id",
            "Topic.Alias": "topic_alias",
            "Title": "title",
            "Body": "idea",
            "idea_type": "idea_type",
            "Publish.Date": "publish_date",
            "Number.of.Votes": "n_votes",
            "Status(selectedbyexpert)": "expert_selected",
            "prior_experience(idea generation)": "idea_experience",
        },
        axis=1,
    )
    ideas = ideas.assign(expert_selected=(ideas["expert_selected"] == 1))
    ideas = ideas.assign(idea=ideas["idea"].map(normalize))
    comments = (
        raw_sheet["Comments"]
        .rename(
            {
                "Submission.ID": "submission_id",
                "Topic.Alias": "topic_alias",
                "Comment.ID": "comment_id",
                "Posted.At": "publish_date",
                "Comment": "comment",
                "Number of votes": "n_votes",
            },
            axis=1,
        )
        .drop(["Submission.Title", "Parent.ID", "Root.ID"], axis=1)
    )
    comments = comments.assign(comment=comments["comment"].map(normalize))
    user = raw_sheet["ideator"]
    user = user.assign(location=user["location"].map(clean_country_name))
    return {"comments": comments, "ideas": ideas, "users": user}
