
import streamlit as st
import nltk
nltk.download("punkt")

from summarizers.lexrank import summarize_lexrank
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
use_lexrank = st.sidebar.checkbox("Use LexRank", value=True)

num_sentences = st.sidebar.slider("Number of sentences", 1, 10, 3)

input_source = st.sidebar.radio("Choose input:", ("Sample Text", "Paste your own"))
if input_source == "Paste your own":
    input_text = st.sidebar.text_area("Paste text here:", height=300)
else:
    input_text = default_text

# Main UI
st.title("🧠 Text Summarization Demo (LexRank)")
with st.expander("📄 Show Original Text"):
    st.write(input_text)

if use_lexrank:
    st.subheader("📃 LexRank Summary")
    summary = summarize_lexrank(input_text, num_sentences=num_sentences)
    st.write(summary)
else:
    st.info("Please select at least one summarization method in the sidebar.")
