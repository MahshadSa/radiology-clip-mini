from __future__ import annotations

import re
from typing import Dict, List, Tuple

from transformers import DistilBertTokenizerFast


def select_findings_or_impression(
    record: Dict,
    preferred_fields: Tuple[str, ...] = ("findings", "impression", "report"),
) -> str:
    """
    Return the first non-empty text among Findings → Impression → Report.
    Falls back to "" if none exist.
    """
    for key in preferred_fields:
        value = record.get(key, "")
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return ""

_HEADER_RE = re.compile(r"^\s*(findings?|impressions?)\s*:\s*", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

def clean_clinical_text(text: str, lowercase: bool = True) -> str:
    """
    - strip leading/trailing spaces
    - drop a simple 'Findings:' or 'Impression:' header at the start
    - lowercase 
    - collapse repeated whitespace to a single space
    """
    if not isinstance(text, str):
        return ""

    s = text.strip()
    s = _HEADER_RE.sub("", s)            
    if lowercase:
        s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)      
    return s.strip()



def get_distilbert_tokenizer(model_name: str = "distilbert-base-uncased") -> DistilBertTokenizerFast:
    """
    Load DistilBertTokenizerFast. With default uncased English model.
    """
    return DistilBertTokenizerFast.from_pretrained(model_name)


def tokenize_texts(
    texts: List[str],
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 128,
    return_tensors: str = "pt",  
):
    """
    Tokenize a list of strings with padding/truncation for batching.
    Returns a dict with input_ids, attention_mask and optionally tensors.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )


def extract_and_clean_texts_from_records(
    records: List[Dict],
    lowercase: bool = True,
) -> List[str]:
    """
    Convenience: for a list of dataset records, select the preferred field and clean it.
    """
    out: List[str] = []
    for r in records:
        raw = select_findings_or_impression(r)
        out.append(clean_clinical_text(raw, lowercase=lowercase))
    return out
