import spacy
import pytextrank
import os
from spacy.util import get_package_path
from pathlib import Path

def load_pipeline():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm", False)
        model_path = get_package_path("en_core_web_sm")
        nlp = spacy.load(model_path)
    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe("textrank")
    return nlp
