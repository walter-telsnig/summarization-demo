
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
    default_text = "The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test
                    as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed
                    that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC
report in 1966, which found that ten years of research had failed to fulfill the expectations, funding for machine translation was dramatically
reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation
systems were developed. Some of the earliest-used NLP algorithms were based on symbolic methods, such as rewriting grammars or sets of production rules. In the
late 1980s and mid-1990s, the NLP community began to shift toward machine learning, particularly statistical methods. More recently,
deep learning methods, including transformer-based architectures like BERT and GPT, have become state of the art in many NLP tasks."






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
