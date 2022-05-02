import re
from typing import Iterable, List
import spacy


def normalize(text: str) -> str:
    """Normalize text :))"""
    result = re.sub("<[a][^>]*>(.+?)</[a]>", "Link.", text)
    result = re.sub("&gt;", "", result)
    result = re.sub("&#x27;", "'", result)
    result = re.sub("&apos;", "'", result)
    result = re.sub("&quot;", '"', result)
    result = re.sub("&#x2F;", " ", result)
    result = re.sub("<p>", " ", result)
    result = re.sub("</i>", "", result)
    result = re.sub("&#62;", "", result)
    result = re.sub("<i>", " ", result)
    result = re.sub("\n", "", result)
    ascii_encoded = result.encode("ascii", "ignore")
    return ascii_encoded.decode()


nlp = spacy.load("en_core_web_sm")


def lemmatize_texts(texts: Iterable[str]) -> Iterable[List[str]]:
    for doc in nlp.pipe(texts):
        yield [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
