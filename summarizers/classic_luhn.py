
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from nltk.tokenize import sent_tokenize

def summarize_luhn(text, num_sentences=3):
    # Use NLTK's sent_tokenize directly (requires only punkt)
    parser = PlaintextParser.from_string(text, TokenizerWrapper())
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

class TokenizerWrapper:
    def __call__(self, text):
        return sent_tokenize(text)
