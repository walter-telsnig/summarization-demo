
import spacy
import pytextrank
import os
import tarfile
import urllib.request
from pathlib import Path

MODEL_NAME = "en_core_web_sm"
MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
LOCAL_DIR = Path("spacy_model")
EXTRACTED_PATH = LOCAL_DIR / MODEL_NAME

def download_and_extract_model():
    LOCAL_DIR.mkdir(exist_ok=True)
    tar_path = LOCAL_DIR / f"{MODEL_NAME}.tar.gz"

    if not EXTRACTED_PATH.exists():
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, tar_path)
        print("Extracting model...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=LOCAL_DIR)
        os.remove(tar_path)

def load_pipeline():
    download_and_extract_model()
    nlp = spacy.load(EXTRACTED_PATH)
    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe("textrank")
    return nlp

def summarize_spacy_textrank(text, limit_phrases=10):
    nlp = load_pipeline()
    doc = nlp(text)
    return " ".join(str(s) for s in doc._.textrank.summary(limit_phrases=limit_phrases))
