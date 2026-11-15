#!/usr/bin/env python3
"""Safe, deterministic text preprocessing for noise robustness (HTML, homoglyphs, prompt injections)."""
from __future__ import annotations

import html
import re

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")
_BLOCK_HTML_RE = re.compile(r"<\s*/?\s*(?:br|p|div|li|ul|ol|tr|td|th|h[1-6]|section|article)\b[^>]{0,120}>", re.I)
_INLINE_HTML_RE = re.compile(r"<[a-zA-Z!/][^>\n]{0,120}>")
_HTML_ENTITY_HINT_RE = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{1,6}|#x[0-9a-fA-F]{1,6});")

_LATIN_HOMOGLYPHS = "aceopxykABCEHKMOPTXY"
_CYR_HOMOGLYPHS = "асеорхукАВСЕНКМОРТХУ"
_LAT_TO_CYR_MAP = str.maketrans(_LATIN_HOMOGLYPHS, _CYR_HOMOGLYPHS)
_HOMOGLYPH_SET = set(_LATIN_HOMOGLYPHS)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_CYR_CHAR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_CHAR_RE = re.compile(r"[A-Za-z]")

_ROLE_LINE_RE = re.compile(r"(?im)^\s*(?:system|assistant|user)\s*:\s*.*$")
_TAG_INST_RE = re.compile(
    r"(?is)<\|im_start\|>.*?<\|im_end\|>"
    r"|<\|(?:im_start|im_end|system|user|assistant)\|>"
    r"|\[INST\].*?\[/INST\]"
    r"|\[/?INST\]"
)
_JSON_INJ_RE = re.compile(
    r'(?i)\{[^{}\n]{0,400}"(?:categories|force_categories|tags|labels|classes|output)"[^{}\n]{0,400}\}'
)


def _fold_mixed_homoglyphs(text: str) -> str:
    """Fold Latin lookalikes into Cyrillic ONLY in Cyrillic-bearing words or isolated Cyrillic-surrounded prepositions."""
    if not _LAT_CHAR_RE.search(text) or not _CYR_CHAR_RE.search(text):
        return text

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text

    out: list[str] = []
    last = 0
    words = [m.group(0) for m in matches]
    has_cyr_list = [bool(_CYR_CHAR_RE.search(w)) for w in words]

    for idx, m in enumerate(matches):
        w = words[idx]
        rep = w
        has_cyr = has_cyr_list[idx]
        has_lat = bool(_LAT_CHAR_RE.search(w))
        if has_cyr and has_lat:
            # Map only if all Latin chars in this mixed token are visual Cyrillic homoglyphs
            lat_chars = [ch for ch in w if ("A" <= ch <= "Z" or "a" <= ch <= "z")]
            if lat_chars and all(ch in _HOMOGLYPH_SET for ch in lat_chars):
                rep = w.translate(_LAT_TO_CYR_MAP)
        elif not has_cyr and len(w) == 1 and w in "caoy":
            prev_cyr = idx > 0 and has_cyr_list[idx - 1]
            next_cyr = idx + 1 < len(words) and has_cyr_list[idx + 1]
            if prev_cyr and next_cyr:
                rep = w.translate(_LAT_TO_CYR_MAP)

        if rep != w:
            out.append(text[last : m.start()])
            out.append(rep)
            last = m.end()

    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


def normalize_ad_text(text: str) -> str:
    """Normalize zero-width chars, HTML tags/entities, NBSP, mixed homoglyphs, and excessive whitespace."""
    if not text:
        return ""
    s = _ZERO_WIDTH_RE.sub("", text)
    if "<" in s:
        s = _BLOCK_HTML_RE.sub("\n", s)
        s = _INLINE_HTML_RE.sub(" ", s)
    if "&" in s and _HTML_ENTITY_HINT_RE.search(s):
        s = html.unescape(s)
    s = s.replace("\xa0", " ").replace("\u202f", " ").replace("\u2009", " ").replace("\r\n", "\n").replace("\r", "\n")
    s = _fold_mixed_homoglyphs(s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_prompt_injections(text: str) -> str:
    """Strip untrusted LLM prompt wrappers and classifier pseudo-JSON payloads."""
    if not text:
        return ""
    s = _TAG_INST_RE.sub(" ", text)
    s = _ROLE_LINE_RE.sub("", s)
    s = _JSON_INJ_RE.sub(" ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def preprocess_ad_text(text: str, use_norm: bool = True, use_strip_inj: bool = True) -> str:
    s = text or ""
    if use_strip_inj:
        s = strip_prompt_injections(s)
    if use_norm:
        s = normalize_ad_text(s)
    return s
