
from transformers import pipeline
import torch

# Load models only once
@torch.inference_mode()
def summarize_bart(text, max_length=100):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return result[0]["summary_text"]

@torch.inference_mode()
def summarize_t5(text, max_length=100):
    summarizer = pipeline("summarization", model="t5-base")
    result = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return result[0]["summary_text"]
