
from gensim.summarization import summarize

def summarize_textrank(text, ratio=0.2):
    try:
        return summarize(text, ratio=ratio)
    except ValueError:
        return "Text too short for TextRank summarization."
