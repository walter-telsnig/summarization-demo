
import streamlit as st
import nltk
nltk.download("punkt")

from summarizers.textrank_spacy import summarize_spacy_textrank
from pathlib import Path

# Load sample text
try:
    default_text = Path("data/sample_text.txt").read_text()
except FileNotFoundError:
    default_text = """
    The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods.
    In 1950, Alan Turing published an article titled 'Computing Machinery and Intelligence' which proposed what is now called the Turing test.
    More recently, transformer-based architectures like BERT and GPT have become state of the art in many NLP tasks.
    """.strip()

# Sidebar
st.sidebar.title("🔧 Summarization Settings")
use_textrank = st.sidebar.checkbox("Use spaCy TextRank", value=True)

num_phrases = st.sidebar.slider("Summary size (# key phrases)", 1, 20, 10)

input_source = st.sidebar.radio("Choose input:", ("Sample Text", "Paste your own"))
if input_source == "Paste your own":
    input_text = st.sidebar.text_area("Paste text here:", height=300)
else:
    input_text = default_text

# Main UI
st.title("🧠 Text Summarization Demo (spaCy TextRank)")
with st.expander("📄 Show Original Text"):
    st.write(input_text)

if use_textrank:
    st.subheader("📃 Extractive Summary (TextRank)")
    summary = summarize_spacy_textrank(input_text, limit_phrases=num_phrases)
    st.write(summary)
else:
    st.info("Please select at least one summarization method in the sidebar.")
