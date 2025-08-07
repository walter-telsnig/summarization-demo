
import spacy
import pytextrank

# Load spaCy English pipeline and add PyTextRank
def load_pipeline():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    if not any(isinstance(pipe[1], pytextrank.TextRank) for pipe in nlp.pipeline):
        nlp.add_pipe("textrank")
    return nlp

def summarize_spacy_textrank(text, limit_phrases=10):
    nlp = load_pipeline()
    doc = nlp(text)
    return " ".join([str(s) for s in doc._.textrank.summary(limit_phrases=limit_phrases)])
