
import streamlit as st
import nltk
nltk.download("punkt")

from summarizers.classic_luhn import summarize_luhn
from summarizers.huggingface import summarize_bart, summarize_t5
from pathlib import Path

# Load sample text
try:
    default_text = Path("data/sample_text.txt").read_text()
except FileNotFoundError:
    default_text = "The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. in 1950, Alan Turing published an article titled Computing Machinery and Intelligence which proposed what is now called the Turing test"

# Sidebar UI
st.sidebar.title("🔧 Summarization Settings")
use_luhn = st.sidebar.checkbox("Use Luhn", value=True)
use_bart = st.sidebar.checkbox("Use BART", value=True)
use_t5 = st.sidebar.checkbox("Use T5", value=False)

summary_length = st.sidebar.slider("Summary length (#sentences for Luhn, #words for BART/T5)", 30, 150, 80)

input_source = st.sidebar.radio("Choose input:", ("Sample Text", "Paste your own"))
if input_source == "Paste your own":
    input_text = st.sidebar.text_area("Paste text here:", height=300)
else:
    input_text = default_text

# Main UI
st.title("🧠 Text Summarization Showdown")
with st.expander("📄 Show Original Text"):
    st.write(input_text)

columns = []
models = []

if use_luhn:
    models.append(("Luhn", summarize_luhn(input_text, num_sentences=3)))
if use_bart:
    models.append(("BART", summarize_bart(input_text, max_length=summary_length)))
if use_t5:
    models.append(("T5", summarize_t5(input_text, max_length=summary_length)))

if models:
    st.subheader("📃 Summaries")
    cols = st.columns(len(models))
    for col, (name, summary) in zip(cols, models):
        with col:
            st.markdown(f"### {name}")
            st.write(summary)
else:
    st.info("Please select at least one summarization method in the sidebar.")
