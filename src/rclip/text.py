from __future__ import annotations

import re
from typing import Dict, List, Tuple

from transformers import DistilBertTokenizerFast


def select_findings_or_impression(
    record: Dict,
    preferred_fields: Tuple[str, ...] = ("findings", "impression", "report", "report_text"),
) -> str:
    """
    Return the first non-empty text among Findings/Impression/Report (case-insensitive).
    Falls back to "" if none exist.
    """
    # build a case-insensitive view of the record
    lower_map = {str(k).lower(): k for k in record.keys()}
    for key in preferred_fields:
        if key.lower() in lower_map:
            v = record.get(lower_map[key.lower()], "")
            if isinstance(v, str):
                v = v.strip()
                if v:
                    return v
    return ""


_HEADER_RE = re.compile(r"^\s*(findings?|impressions?)\s*:\s*", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_clinical_text(text: str, lowercase: bool = True) -> str:
    """
    - strip leading/trailing spaces
    - drop a 'Findings:' or 'Impression:' header at the start
    - optional lowercase
    - collapse repeated whitespace
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
    return DistilBertTokenizerFast.from_pretrained(model_name)


def tokenize_texts(
    texts: List[str],
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 128,
    return_tensors: str = "pt",
):
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
    out: List[str] = []
    for r in records:
        raw = select_findings_or_impression(r)
        out.append(clean_clinical_text(raw, lowercase=lowercase))
    return out
