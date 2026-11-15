#!/usr/bin/env python3
"""Fixed semantic chunker for Category-Aware Qwen3 experiments (preserves short lines, headings, lists, hashtags)."""
import re

BULLET_RE = re.compile(
    r"^(?:[•·▪▫◾◽■□●○◆◇◉➤▶▸►✔✅☑❌❎🔹🔸🔶🔷➕➖📌🔧🔨⚡🚿🎨🧱🏠🏡📲📞📸⭐🌟🔥🎁📍❗]\ufe0f?\s*|"
    r"(?:[\-–—+]|(?<!\*)\*(?!\*)|\d+[.)])\s+)"
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«\"])")
SENT_FALLBACK_RE = re.compile(
    r"(?<=[.!?])(?:\s+|(?=[\U0001f300-\U0001faff✅✔🔹🔸🔧🔨⚡🚿🎨🧱🏠🏡📲📞📸⭐🔥🎁📍❗🛠📐📋💥💬💡🔎])|(?=[А-ЯЁA-Z]))"
)
SHORT_ANSWER = re.compile(
    r"^(?:да|нет|верно|конечно|отвечаем|услуга\s+(?:доступна|не)|только\s+|однако\b|но\b)",
    re.I,
)


def split_long_line(ln: str, max_chars: int = 420) -> list[str]:
    """Split long single lines by sentences, falling back to emoji/uppercase boundaries and words only if needed."""
    out = []
    for part in (s.strip() for s in SENT_SPLIT.split(ln) if s.strip()):
        if len(part) <= max_chars:
            out.append(part)
            continue
        for sub in (s.strip() for s in SENT_FALLBACK_RE.split(part) if s.strip()):
            if len(sub) <= max_chars:
                out.append(sub)
            else:
                words, cur, cl = sub.split(), [], 0
                for w in words:
                    if cur and cl + 1 + len(w) > max_chars:
                        out.append(" ".join(cur))
                        cur, cl = [w], len(w)
                    else:
                        cl += (1 if cur else 0) + len(w)
                        cur.append(w)
                if cur:
                    out.append(" ".join(cur))
    return out


def is_hashtag_line(ln: str) -> bool:
    if ln.startswith("#"):
        return True
    toks = ln.split()
    return bool(toks) and sum(1 for t in toks if t.startswith("#")) >= max(1, len(toks) // 2)


def is_short_unpunctuated(ln: str) -> bool:
    if BULLET_RE.match(ln) or is_hashtag_line(ln):
        return False
    if ln.endswith((":", ".", "!", "?", ",", ";")):
        return False
    return len(ln) <= 55 and len(ln.split()) <= 7


def chunk_ad_fixed(text: str, target_chars: int = 300, max_chars: int = 420) -> list[str]:
    """Segment full ad text without losing short lines, consecutive headings, lists, or hashtags."""
    text = text.strip()
    if not text:
        return [""]

    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    raw_units = []
    for p_idx, p in enumerate(raw_paras):
        if p_idx > 0:
            raw_units.append(("break", ""))
        for raw_ln in p.splitlines():
            ln = re.sub(r"[ \t]+", " ", raw_ln).strip()
            if not ln or re.fullmatch(r"[\W_]{3,}", ln):
                continue
            if BULLET_RE.match(ln):
                if len(ln) <= 240:
                    raw_units.append(("bullet", ln))
                else:
                    for s in split_long_line(ln, max_chars=max_chars):
                        raw_units.append(("prose", s))
            elif ln.endswith(":") and len(ln) <= 90:
                raw_units.append(("colon_header", ln))
            elif is_hashtag_line(ln):
                if len(ln) <= max_chars:
                    raw_units.append(("list_item", ln))
                else:
                    for s in split_long_line(ln, max_chars=max_chars):
                        raw_units.append(("prose", s))
            elif is_short_unpunctuated(ln):
                raw_units.append(("short_cand", ln))
            elif len(ln) <= 240:
                raw_units.append(("prose", ln))
            else:
                for s in split_long_line(ln, max_chars=max_chars):
                    raw_units.append(("prose", s))

    # A short unpunctuated line is a heading ONLY if isolated and immediately followed by bullet/prose
    # (never if it's part of a sequence of short lines or before EOF/colon_header/hashtag).
    units = []
    n_u = len(raw_units)
    for idx, (k, v) in enumerate(raw_units):
        if k == "colon_header":
            units.append(("header", v))
        elif k == "short_cand":
            nxt_k = next((raw_units[j][0] for j in range(idx + 1, n_u) if raw_units[j][0] != "break"), None)
            prev_k = next((raw_units[j][0] for j in range(idx - 1, -1, -1) if raw_units[j][0] != "break"), None)
            if nxt_k in ("bullet", "prose") and prev_k != "short_cand":
                units.append(("header", v))
            else:
                units.append(("list_item", v))
        else:
            units.append((k, v))

    chunks = []
    cur_header = None
    header_pending = False
    header_used = False
    i = 0
    while i < len(units):
        kind, val = units[i]
        if kind == "break":
            if not header_pending:
                cur_header = None
                header_used = False
            i += 1
            continue
        if kind == "header":
            if header_pending and cur_header is not None:
                chunks.append(cur_header)
            cur_header = val
            header_pending = True
            header_used = False
            i += 1
            continue
        header_pending = False
        if kind == "bullet":
            block = val
            while (
                i + 1 < len(units)
                and units[i + 1][0] == "prose"
                and (block.endswith("?") or SHORT_ANSWER.match(units[i + 1][1]) or len(units[i + 1][1]) < 45)
                and len(block) + len(units[i + 1][1]) + 1 <= max_chars
            ):
                i += 1
                block = block + " " + units[i][1]
            if cur_header and len(cur_header) + len(block) + 1 <= max_chars:
                chunks.append(f"{cur_header}\n{block}")
                header_used = True
            else:
                if cur_header and not header_used:
                    chunks.append(cur_header)
                    header_used = True
                chunks.append(block)
            i += 1
            continue
        if kind == "list_item":
            if cur_header and len(cur_header) + len(val) + 1 <= max_chars:
                chunks.append(f"{cur_header}\n{val}")
                header_used = True
            else:
                if cur_header and not header_used:
                    chunks.append(cur_header)
                    header_used = True
                chunks.append(val)
            i += 1
            continue

        buf = [val]
        prefix = f"{cur_header}\n" if cur_header else ""
        cur_len = len(prefix) + len(val)
        j = i + 1
        while j < len(units) and units[j][0] == "prose":
            nxt = units[j][1]
            must_attach = buf[-1].endswith("?") or SHORT_ANSWER.match(nxt) or len(nxt) < 55
            if cur_len + 1 + len(nxt) <= (max_chars if must_attach else target_chars):
                buf.append(nxt)
                cur_len += 1 + len(nxt)
                j += 1
            else:
                break
        chunks.append(prefix + " ".join(buf))
        cur_header = None
        header_used = False
        if j < len(units) and units[j][0] == "prose" and len(buf) >= 2:
            i = j - 1
        else:
            i = j

    if header_pending and cur_header:
        chunks.append(cur_header)
    return chunks or [text[:max_chars]]
