"""
preprocessor.py
---------------
Text cleaning and normalization pipeline for resumes and job descriptions.
Handles PDF extraction, noise removal, lemmatization, and stopword filtering.
"""

import re
import string
import logging
from pathlib import Path
from typing import Union

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (silent)
for pkg in ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

logger = logging.getLogger(__name__)


def load_spacy_model():
    """Load spaCy model, falling back gracefully."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return None


nlp = load_spacy_model()
STOP_WORDS = set(stopwords.words("english"))

# Words to KEEP even if stopwords (important for resumes)
KEEP_WORDS = {
    "c", "r", "go", "ai", "ml", "dl", "bi", "qa", "ui", "ux",
    "no", "not", "above", "below", "between", "with", "without"
}
EFFECTIVE_STOP_WORDS = STOP_WORDS - KEEP_WORDS


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Extract raw text from a PDF file using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
        return ""


def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """Auto-detect file type and extract text."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in (".txt", ".md", ".text"):
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix in (".docx",):
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.error("python-docx not installed.")
            return ""
    else:
        # Try reading as plain text as fallback
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove URLs, emails, phone numbers
    3. Remove special characters (keep hyphens in compound words)
    4. Normalize whitespace
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", " ", text)

    # Remove special chars but keep hyphen between words (e.g. "full-stack")
    text = re.sub(r"[^\w\s\-\+\#\.]", " ", text)

    # Remove lone numbers (page numbers, dates as standalone)
    text = re.sub(r"\b\d{1,2}\b", " ", text)

    # Remove extra dots (e.g. ".........")
    text = re.sub(r"\.{2,}", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_filter(text: str) -> list[str]:
    """Tokenize and remove stopwords."""
    tokens = word_tokenize(text)
    return [
        t for t in tokens
        if t not in EFFECTIVE_STOP_WORDS
        and len(t) > 1
        and not t.isnumeric()
    ]


def lemmatize(text: str) -> str:
    """Lemmatize text using spaCy if available, else return cleaned text."""
    if nlp is None:
        return text
    doc = nlp(text[:100000])  # spaCy max length guard
    return " ".join(
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 1
    )


def preprocess(text: str, lemmatize_text: bool = True) -> dict:
    """
    Full preprocessing pipeline. Returns a dict with:
    - raw: original text
    - cleaned: cleaned text
    - lemmatized: lemmatized text
    - tokens: filtered token list
    """
    raw = text.strip()
    cleaned = clean_text(raw)

    if lemmatize_text:
        lemmatized = lemmatize(cleaned)
    else:
        lemmatized = cleaned

    tokens = tokenize_and_filter(lemmatized)

    return {
        "raw": raw,
        "cleaned": cleaned,
        "lemmatized": lemmatized,
        "tokens": tokens,
    }


def preprocess_file(file_path: Union[str, Path], lemmatize_text: bool = True) -> dict:
    """Extract text from file and run full preprocessing pipeline."""
    raw_text = extract_text_from_file(file_path)
    result = preprocess(raw_text, lemmatize_text)
    result["source"] = str(file_path)
    return result
