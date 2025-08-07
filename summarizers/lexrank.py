from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.tokenize import sent_tokenize, word_tokenize

class NLTKTokenizer:
    def to_sentences(self, text):
        return sent_tokenize(text)

    def to_words(self, sentence):
        return word_tokenize(sentence)

def summarize_lexrank(text, num_sentences=3):
    tokenizer = NLTKTokenizer()
    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
