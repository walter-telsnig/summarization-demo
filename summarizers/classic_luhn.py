from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from nltk.tokenize import sent_tokenize

def summarize_luhn(text, num_sentences=3):
    # Use sentence tokenizer from nltk directly
    sentences = sent_tokenize(text)
    joined_text = " ".join(sentences)

    parser = PlaintextParser.from_string(joined_text, lambda txt: sent_tokenize(txt))
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
