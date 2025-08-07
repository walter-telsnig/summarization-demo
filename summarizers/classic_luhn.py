from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from nltk.tokenize import sent_tokenize, word_tokenize

class CustomTokenizer:
    def to_sentences(self, text):
        return sent_tokenize(text)
    
    def to_words(self, sentence):
        return word_tokenize(sentence)

def summarize_luhn(text, num_sentences=3):
    tokenizer = CustomTokenizer()
    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
