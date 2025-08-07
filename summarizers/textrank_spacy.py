
import spacy
import pytextrank
from spacy.cli import download
from spacy.util import is_package, get_package_path

def load_pipeline():
    model = "en_core_web_sm"
    try:
        nlp = spacy.load(model)
    except OSError:
        if not is_package(model):
            download(model)  # Downloads into user space (safe on Streamlit Cloud)
        model_path = get_package_path(model)
        nlp = spacy.load(model_path)

    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe("textrank")
    return nlp

def summarize_spacy_textrank(text, limit_phrases=10):
    nlp = load_pipeline()
    doc = nlp(text)
    return " ".join([str(s) for s in doc._.textrank.summary(limit_phrases=limit_phrases)])
