#!/usr/bin/env python3
"""Russian UD dependency-tree feature extractor (Natasha / Slovnet) for Category-Aware Avito classification."""
from __future__ import annotations

import inspect
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Segmenter

# General category lexical stems (strictly grounded in category definitions 102..106, no dataset-ID tuning).
# Uses normalized word prefixes/stems to survive lemmatizer ambiguity (e.g. "электрику" -> "электрик").
CATEGORY_PATTERNS: dict[int, re.Pattern[str]] = {
    102: re.compile(
        r"^(?:штукатур\w*|оштукатур\w*|маяк\w*|выравниван\w*|откос\w*|стяжк\w*|ровн\w*)",
        re.I,
    ),
    103: re.compile(
        r"^(?:плитк\w*|плиточ\w*|кафел\w*|керамогранит\w*|мозаик\w*|затир\w*|перезатир\w*|облицов\w*|запил\w*|фартук\w*)",
        re.I,
    ),
    104: re.compile(
        r"^(?:сантех\w*|водоснаб\w*|водопровод\w*|канализац\w*|отоплен\w*|труб\w*|раковин\w*|унитаз\w*|умывальн\w*|"
        r"инсталляц\w*|ванн\w*|душев\w*|смесител\w*|сифон\w*|протеч\w*|теч[ьи]\w*|засор\w*|радиатор\w*|"
        r"водонагревател\w*|бойлер\w*|полотенцесуш\w*|мойк\w*|посудомо\w*|стиральн\w*|слив\w*|фильтр\w*)",
        re.I,
    ),
    105: re.compile(
        r"^(?:электр\w*|электромонтаж\w*|проводк\w*|кабел\w*|розет\w*|выключател\w*|люстр\w*|светильник\w*|"
        r"электроточ\w*|электрощит\w*|щит|щита|щиту|щитом|щите|щитк\w*|автомат\w*|узо|дифавтомат\w*|штробл\w*|подсветк\w*|нагревател\w*)",
        re.I,
    ),
    106: re.compile(
        r"^(?:маляр\w*|обо[ийея]\w*|обойн\w*|поклей\w*|оклей\w*|наклеив\w*|клеи\w*|крас\w*|покрас\w*|перекрас\w*|окрас\w*|"
        r"окрашив\w*|шпакл\w*|шпатл\w*|грунтов\w*|шлифов\w*|выкрас\w*|стеклохолст\w*)",
        re.I,
    ),
}

# Disambiguators where a generic stem only signals category when paired in phrase/context
WEAK_MOD_WORDS = re.compile(r"^(?:ровн\w*|фильтр\w*|автомат\w*|нагревател\w*|клеи\w*|слив\w*)$", re.I)

ANAPHORA_NOUN_RE = re.compile(r"^(?:работ\w*|услуг\w*|заказ\w*|заявк\w*|задач\w*|дел[оауе]\w*|операц\w*|позиц\w*|эт[оауи]\w*|перечислен\w*|такой|такую|такие|такого|данный|рамк\w*|он|она|оно|они|его|её|их)$", re.I)
SERVICE_WORK_NOUN_RE = re.compile(r"^(?:работ\w*|услуг\w*|заказ\w*|заявк\w*|операц\w*|додел\w*|комнат\w*|этап\w*|вид\w*|ремонт\w*|проект\w*|объект\w*)$", re.I)
QUERY_META_LEMMAS = {"запрос", "фраза", "ключевик", "тег", "слово", "поиск", "объявление", "заголовок"}
NON_SERVICE_NEG_OBJ_RE = re.compile(
    r"^(?:предоплат\w*|доплат\w*|аванс\w*|денег|деньг\w*|наценк\w*|посредник\w*|комисси\w*|процент\w*|переплат\w*|"
    r"оттен\w*|результат\w*|ошиб\w*|брак\w*|мусор\w*|шум\w*|время|секрет\w*)$",
    re.I,
)

ACTION_LEMMAS = {
    "выполнять", "выполнить", "делать", "сделать", "оказывать", "оказать", "предлагать", "предложить",
    "брать", "взять", "браться", "взяться", "принимать", "принять", "работать", "устанавливать",
    "установить", "укладывать", "уложить", "монтировать", "смонтировать", "клеить", "наклеить",
    "поклейка", "красить", "покрасить", "штукатурить", "оштукатурить", "шпаклевать", "заменять",
    "заменить", "менять", "поменять", "подключать", "подключить", "прокладывать", "проложить",
    "собирать", "собрать", "заниматься", "заняться", "помогать", "помочь", "заказывать", "заказать",
    "нанимать", "нанять", "приезжать", "приехать", "проводить", "провести", "производить", "осуществлять",
    "вести", "предоставлять", "предоставить", "реализовывать", "мочь", "смочь", "быть", "являться",
    "входить", "действовать", "иметься", "нет", "отказывать", "отказать", "отказываться", "отказаться",
    "отклонять", "исключать", "исключить", "отменять", "прекращать", "прекратить", "переставать",
    "перестать", "получать", "получить", "продолжать", "оставаться", "обещать", "гарантировать",
    "согласовывать", "добавлять", "добавить", "исключено", "невозможно", "нужный", "нужно", "надо",
    "возможно", "возможный", "доступный", "доступно", "готовый", "готов", "хотеть",
    "отвечать", "ответить", "устареть", "устаревать",
}
REFUSAL_ACTION_LEMMAS = {
    "отказываться", "отказаться", "отказывать", "отказать", "отклонять", "отклонить",
    "исключать", "исключить", "отменять", "отменить", "прекращать", "прекратить",
    "переставать", "перестать", "устареть", "устаревать",
}
REFUSAL_NOUN_LEMMAS = {"отказ", "исключение", "недоступность", "рамка"}
INQUIRY_LEMMAS = {"спрашивать", "спросить", "просить", "попросить", "уточнять", "уточнить", "интересоваться"}
CLIENT_WISH_LEMMAS = {"хотеть", "планировать", "собираться", "пытаться", "намереваться"}
MODAL_NEG_LEMMAS = {"нужный", "нужно", "надо", "стоит", "следовать", "мочь", "смочь", "нельзя"}
PRICE_MOD_LEMMAS = {"бесплатно", "даром", "безвозмездно"}

THIRD_PARTY_LEMMAS = {"подрядчик", "субподрядчик", "субподряд", "сторонний", "чужой", "другой", "соседний", "привлечённый", "привлеченный", "партнёр", "партнер"}
CLIENT_LEMMAS = {"заказчик", "клиент", "владелец", "собственник", "хозяин", "покупатель", "сами", "самостоятельно"}
SELLER_POSSESSIVE_LEMMAS = {"наш", "мой", "свой", "штатный", "личный", "частный"}
SELLER_NOUN_LEMMAS = {"мы", "я", "бригада", "мастер", "команда", "специалист", "руководитель", "исполнитель"}

TEMPORAL_PAST_LEMMAS = {"раньше", "ранее", "прежде", "прошлое", "прошлый", "старый", "прежний"}
TEMPORAL_STOP_WORDS = {"больше", "уже"}
TEMPORAL_NOW_LEMMAS = {"теперь", "сейчас", "ныне", "настоящий", "актуальный", "снова", "вновь"}

PARTIAL_MOD_RE = re.compile(r"^(?:отдельн\w*|частич\w*|мелк\w*|одиноч\w*|разов\w*|локальн\w*|небольш\w*)$", re.I)
GLOBAL_CANDIDATE_RE = re.compile(
    r"(?:под\s+ключ|целиком|комплекс|отдельн\w*|частич\w*|мелк\w*|одиноч\w*|только\s+)",
    re.I,
)


@dataclass
class ParsedToken:
    id: str
    text: str
    lemma: str
    pos: str
    head_id: str
    rel: str
    feats: dict[str, str]
    idx: int  # 0-based position inside sentence


@dataclass
class ParsedSentence:
    text: str
    tokens: list[ParsedToken]
    by_id: dict[str, ParsedToken]
    children: dict[str, list[ParsedToken]]
    undirected: dict[str, list[str]]


class SyntaxAnalyzer:
    """Thread-safe / cached wrapper around Natasha Segmenter + NewsMorphTagger + NewsSyntaxParser."""

    def __init__(self) -> None:
        self.segmenter = Segmenter()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.syntax_parser = NewsSyntaxParser(emb)
        self.morph_vocab = MorphVocab()
        self._sent_cache: dict[str, list[ParsedSentence]] = {}

    def parse_text(self, text: str) -> list[ParsedSentence]:
        key = text.strip()
        if not key:
            return []
        if key in self._sent_cache:
            return self._sent_cache[key]

        doc = Doc(key)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        for tok in doc.tokens:
            tok.lemmatize(self.morph_vocab)

        sents: list[ParsedSentence] = []
        for sent in doc.sents:
            ptoks: list[ParsedToken] = []
            by_id: dict[str, ParsedToken] = {}
            children: dict[str, list[ParsedToken]] = {}
            undirected: dict[str, list[str]] = {}
            for i, t in enumerate(sent.tokens):
                pt = ParsedToken(
                    id=t.id,
                    text=t.text,
                    lemma=(t.lemma or t.text).lower(),
                    pos=t.pos or "",
                    head_id=t.head_id or "",
                    rel=t.rel or "",
                    feats=dict(t.feats or {}),
                    idx=i,
                )
                ptoks.append(pt)
                by_id[pt.id] = pt
                children[pt.id] = []
                undirected[pt.id] = []

            for pt in ptoks:
                hid = pt.head_id
                if hid in by_id and hid != pt.id:
                    children[hid].append(pt)
                    undirected[hid].append(pt.id)
                    undirected[pt.id].append(hid)

            sents.append(
                ParsedSentence(
                    text=sent.text,
                    tokens=ptoks,
                    by_id=by_id,
                    children=children,
                    undirected=undirected,
                )
            )
        self._sent_cache[key] = sents
        return sents


def _shortest_ud_dist(sent: ParsedSentence, src_id: str, dst_id: str) -> int:
    if src_id == dst_id:
        return 0
    if src_id not in sent.by_id or dst_id not in sent.by_id:
        return -1
    q = deque([(src_id, 0)])
    seen = {src_id}
    while q:
        cur, d = q.popleft()
        for nxt in sent.undirected.get(cur, ()):
            if nxt == dst_id:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return -1


def _is_predicate_like(tok: ParsedToken, sent: ParsedSentence) -> bool:
    if tok.lemma in ACTION_LEMMAS or tok.lemma in REFUSAL_ACTION_LEMMAS:
        return True
    if tok.pos in ("VERB", "AUX") and tok.rel not in ("amod", "det", "nmod", "obl", "iobj", "obj", "nsubj", "nsubj:pass"):
        return True
    if tok.pos in ("VERB", "AUX") and any(ch.rel in ("obj", "nsubj", "nsubj:pass", "xcomp", "ccomp") for ch in sent.children.get(tok.id, ())):
        # Guard against Slovnet mistagging service nouns ("электромонтаж", "шпаклёвкой") as VERB
        if tok.rel in ("obj", "nsubj", "nsubj:pass", "obl", "iobj", "amod") and tok.head_id != tok.id:
            return False
        return True
    if tok.rel in ("root", "conj", "parataxis", "ccomp", "xcomp", "advcl") and any(
        ch.rel in ("obj", "nsubj", "obl") for ch in sent.children.get(tok.id, ())
    ):
        return True
    return False


def _is_neg_particle(tok: ParsedToken, sent: ParsedSentence) -> bool:
    """True for syntactic negation tokens, excluding 'не только ... но и'."""
    if tok.lemma in ("не", "нет", "ни", "нельзя") or tok.feats.get("Polarity") == "Neg":
        # Exclude "не только ... но и" where "только" is fixed/child of "не" or next token is "только"
        kids = sent.children.get(tok.id, ())
        if any(k.lemma == "только" for k in kids):
            return False
        if tok.idx + 1 < len(sent.tokens) and sent.tokens[tok.idx + 1].lemma == "только":
            return False
        if tok.rel == "cc":
            return False
        return True
    return False


def _find_not_only_in_clause(pred: ParsedToken, sent: ParsedSentence) -> bool:
    nodes = [pred]
    if pred.rel == "conj" and pred.head_id in sent.by_id:
        nodes.append(sent.by_id[pred.head_id])
    for node in nodes:
        for ch in sent.children.get(node.id, ()):
            if ch.lemma == "не" and (ch.rel == "cc" or any(k.lemma == "только" for k in sent.children.get(ch.id, ()))):
                return True
            if ch.lemma == "не" and ch.idx + 1 < len(sent.tokens) and sent.tokens[ch.idx + 1].lemma == "только":
                return True
    return False


def _subtree_tokens(root_id: str, sent: ParsedSentence, stop_rels: set[str] | None = None) -> list[ParsedToken]:
    out = []
    q = deque([root_id])
    seen = {root_id}
    while q:
        cur = q.popleft()
        tok = sent.by_id.get(cur)
        if tok is not None:
            out.append(tok)
        for ch in sent.children.get(cur, ()):
            if stop_rels and ch.rel in stop_rels:
                continue
            if ch.id not in seen:
                seen.add(ch.id)
                q.append(ch.id)
    return out


def _detect_predicate_negation(pred: ParsedToken, sent: ParsedSentence) -> tuple[bool, list[ParsedToken], bool, bool]:
    """
    Returns (has_action_neg, neg_tokens, is_price_cond_neg, is_double_neg).
    Checks direct syntactic negation of `pred` (handling 'не бесплатно, а по смете',
    'не берём предоплату', 'отказываемся', 'получили отказ / нет').
    """
    neg_toks: list[ParsedToken] = []
    price_cond = False
    kids = sent.children.get(pred.id, ())

    # Check non-service object guard (e.g. "не берём предоплату / 10 объектов / посредников")
    # Wait: that is handled when deciding if negation applies to a service vs non-service object,
    # whereas here we check whether `pred` itself is negated or only a price adverb is negated.
    for ch in kids:
        if _is_neg_particle(ch, sent):
            # Look closely at "Не бесплатно, а по смете выполняем ...":
            # if "не" precedes a price modifier sibling or child ("не бесплатно") and there's a positive "а по смете"
            if ch.idx + 1 < len(sent.tokens) and sent.tokens[ch.idx + 1].lemma in PRICE_MOD_LEMMAS:
                price_cond = True
            else:
                neg_toks.append(ch)
        elif ch.lemma in PRICE_MOD_LEMMAS:
            # Check if "не" attached to the price modifier ("не бесплатно")
            for gk in sent.children.get(ch.id, ()):
                if _is_neg_particle(gk, sent):
                    price_cond = True
        elif ch.rel in ("obj", "xcomp", "nsubj") and ch.lemma in REFUSAL_NOUN_LEMMAS:
            neg_toks.append(ch)
            # Check if the refusal noun itself has negation?
        elif ch.rel == "advmod" and ch.lemma == "отказ":
            neg_toks.append(ch)

    # Check if predicate itself is a refusal verb (e.g., "отказываемся", "отказались")
    is_refusal_verb = pred.lemma in REFUSAL_ACTION_LEMMAS or pred.lemma == "нет"
    double_neg = False

    if pred.lemma in REFUSAL_ACTION_LEMMAS:
        if len(neg_toks) >= 1:
            # e.g. "не отказываемся от электромонтажа" -> cancelled refusal!
            double_neg = True
            has_action_neg = False
        else:
            has_action_neg = True
            neg_toks.append(pred)
    elif pred.lemma == "нет":
        has_action_neg = True
        neg_toks.append(pred)
    else:
        # Check if parent modal ("не нужно", "не можем") negates this infinitive/csubj complement
        if pred.rel in ("csubj", "xcomp", "ccomp") and pred.head_id in sent.by_id:
            parent = sent.by_id[pred.head_id]
            if parent.lemma in MODAL_NEG_LEMMAS or parent.pos in ("VERB", "ADJ", "ADV"):
                for pk in sent.children.get(parent.id, ()):
                    if _is_neg_particle(pk, sent):
                        neg_toks.append(pk)
        # Double negation only if a negative modal ("не можем") governs a negated xcomp verb ("не выполнять")
        xcomp_neg = [
            gk for ch in kids if ch.rel == "xcomp"
            for gk in sent.children.get(ch.id, ()) if _is_neg_particle(gk, sent)
        ]
        if pred.lemma in MODAL_NEG_LEMMAS and neg_toks and xcomp_neg:
            double_neg = True
            has_action_neg = False
            neg_toks.extend(xcomp_neg)
        else:
            has_action_neg = len(neg_toks) >= 1

    # Handle "Бесплатно <услугу> не делаем; по смете принимаем" or "Не обещаем выполнить ... бесплатно"
    xcomp_price = any(
        c.lemma in PRICE_MOD_LEMMAS
        for k in kids if k.rel == "xcomp"
        for c in sent.children.get(k.id, ())
    )
    if (has_action_neg or neg_toks) and (any(ch.lemma in PRICE_MOD_LEMMAS for ch in kids) or xcomp_price):
        price_cond = True

    return has_action_neg, neg_toks, price_cond, double_neg


def _extract_clause_actor(pred: ParsedToken, sent: ParsedSentence) -> str:
    """Identify actor for a clause predicate from UD subject/oblique subtree and person morphology."""
    # Clause-local subtree: exclude nested conj/parataxis/advcl/ccomp predicates
    clause_stop = {"conj", "parataxis", "ccomp", "advcl", "acl"}
    kids = sent.children.get(pred.id, ())

    # Collect tokens in subject / agentive obliques (keep acl inside nsubj so "исполнитель, которого заказчик нанял" is seen)
    actor_tokens: list[ParsedToken] = []
    for ch in kids:
        if ch.rel in ("nsubj", "nsubj:pass", "obl:agent"):
            actor_tokens.extend(_subtree_tokens(ch.id, sent, stop_rels={"conj", "parataxis", "ccomp", "advcl"}))
        elif ch.rel == "obl":
            sub = _subtree_tokens(ch.id, sent, stop_rels=clause_stop)
            # Look for "у другой бригады", "от подрядчика", "по нашему договору", "самостоятельно"
            if any(t.lemma in THIRD_PARTY_LEMMAS | CLIENT_LEMMAS | {"бригада", "мастер", "организация", "договор"} for t in sub):
                actor_tokens.extend(sub)

    # Also check adverbials directly on predicate like "выполняет самостоятельно" or "лично"
    for ch in kids:
        if ch.rel == "advmod" and ch.lemma in ("самостоятельно", "лично"):
            actor_tokens.append(ch)

    actor_lemmas = {t.lemma for t in actor_tokens}
    # Ignore "без посредников" / "без подрядчиков"
    without_third = False
    for t in actor_tokens:
        if t.lemma in THIRD_PARTY_LEMMAS | {"посредник"}:
            has_bez = any(
                c.lemma == "без" for c in sent.children.get(t.id, ())
            ) or (t.idx > 0 and sent.tokens[t.idx - 1].lemma == "без")
            if has_bez:
                without_third = True

    # Inquiry verb check ("Клиент спросил про... / Нас попросили"): client is asking, NOT executing!
    if pred.lemma in INQUIRY_LEMMAS:
        return "unknown"

    if not without_third and (actor_lemmas & THIRD_PARTY_LEMMAS):
        return "third_party"
    # "исполнитель, которого заказчик нанял" / "подрядчик заказчика"
    if "исполнитель" in actor_lemmas and (actor_lemmas & CLIENT_LEMMAS or any(t.lemma in ("нанять", "нанимать", "пригласить") for t in _subtree_tokens(pred.id, sent, stop_rels={"conj"}))):
        return "third_party"
    if actor_lemmas & CLIENT_LEMMAS:
        # Careful: "Заказчик заказывает у нас" vs "Заказчик делает сам / у другой бригады"
        # "у нас" -> obl with us?
        has_u_nas = any(
            ch.rel == "obl"
            and any(t.lemma == "мы" for t in _subtree_tokens(ch.id, sent, stop_rels=clause_stop))
            and any(t.lemma == "у" for t in _subtree_tokens(ch.id, sent, stop_rels=clause_stop))
            for ch in kids
        )
        if has_u_nas:
            return "seller_team"
        return "client"

    if actor_lemmas & SELLER_POSSESSIVE_LEMMAS or "мы" in actor_lemmas or "я" in actor_lemmas:
        return "seller_team"
    if actor_lemmas & {"бригада", "мастер", "команда", "руководитель"}:
        return "seller_team"

    # Pro-drop 1st-person Russian verb ("выполняем", "берём", "делаю") -> seller_team!
    if pred.feats.get("Person") == "1":
        return "seller_team"

    return "unknown"


def _extract_clause_temporal(pred: ParsedToken, sent: ParsedSentence, has_neg: bool, double_neg: bool = False) -> str:
    clause_stop = {"conj", "parataxis", "ccomp", "advcl"}
    local_toks = _subtree_tokens(pred.id, sent, stop_rels=clause_stop)
    lems = {t.lemma for t in local_toks}

    # Check "больше не" / "уже не" specifically attached to negation/predicate
    # Note: pymorphy2 lemmatizes "больше" -> "большой" in ADV degree=Cmp, so check text too!
    has_stop = False
    for ch in sent.children.get(pred.id, ()):
        if ch.lemma in TEMPORAL_STOP_WORDS or ch.text.lower() in TEMPORAL_STOP_WORDS:
            has_stop = True
        if ch.lemma == "не":
            for gk in sent.children.get(ch.id, ()):
                if gk.lemma in TEMPORAL_STOP_WORDS or gk.text.lower() in TEMPORAL_STOP_WORDS:
                    has_stop = True
    if has_stop and double_neg:
        return "now"
    if has_stop and has_neg:
        return "stopped"
    if lems & TEMPORAL_NOW_LEMMAS:
        return "now"
    if lems & TEMPORAL_PAST_LEMMAS or (pred.feats.get("Tense") == "Past" and pred.lemma not in INQUIRY_LEMMAS):
        return "past"
    return "unknown"


def _find_governing_predicate(tok: ParsedToken, sent: ParsedSentence) -> tuple[ParsedToken | None, str]:
    """Walk up UD tree (cycle-safe!) to find governing predicate / clause head."""
    if _is_predicate_like(tok, sent) and (tok.lemma in ACTION_LEMMAS | REFUSAL_ACTION_LEMMAS or tok.rel in ("root", "conj", "parataxis")):
        return tok, "self"
    visited = {tok.id}
    cur = tok
    steps = 0
    while cur.head_id and cur.head_id in sent.by_id and cur.head_id not in visited and steps < 12:
        nxt = sent.by_id[cur.head_id]
        visited.add(nxt.id)
        steps += 1
        if _is_predicate_like(nxt, sent):
            rel_kind = "direct_verb_obj" if cur.rel in ("obj", "nsubj", "nsubj:pass") and steps == 1 else "prep_obl_or_nmod"
            return nxt, rel_kind
        cur = nxt
    return None, "orphan_mention"


def _token_matches_category(tok: ParsedToken, cat_id: int) -> bool:
    pat = CATEGORY_PATTERNS[cat_id]
    matched = bool(pat.match(tok.lemma) or pat.match(tok.text))
    if not matched:
        return False
    # Exclude weak single-word modifiers unless linked directly to repair context
    if WEAK_MOD_WORDS.match(tok.lemma) or WEAK_MOD_WORDS.match(tok.text):
        return False
    return True


def _detect_global_restrictions_in_sentence(sent: ParsedSentence) -> tuple[bool, bool, list[str]]:
    """
    Detect (global_turnkey_only, global_partial_refusal, traces) from UD tree of a sentence.
    Requires syntactic grounding (not bare substring 'не').
    """
    if not GLOBAL_CANDIDATE_RE.search(sent.text):
        return False, False, []

    turnkey_only = False
    partial_refusal = False
    traces: list[str] = []

    has_turnkey_phrase = bool(re.search(r"под\s+ключ|целиком|комплекс\w*\s+ремонт", sent.text, re.I))
    if has_turnkey_phrase:
        for tok in sent.tokens:
            if tok.lemma in ("ключ", "целиком", "комплекс", "комплексный"):
                # Check if "только" or "исключительно" modifies the same noun phrase or governing predicate
                pred, _ = _find_governing_predicate(tok, sent)
                check_nodes = [tok]
                if tok.head_id in sent.by_id:
                    check_nodes.append(sent.by_id[tok.head_id])
                if pred is not None:
                    check_nodes.append(pred)
                for node in check_nodes:
                    for ch in sent.children.get(node.id, ()):
                        if ch.lemma in ("только", "исключительно"):
                            # Ensure NOT "не только под ключ"
                            if not any(k.lemma == "не" for k in sent.children.get(ch.id, ())):
                                not_only = pred is not None and _find_not_only_in_clause(pred, sent)
                                if not not_only:
                                    turnkey_only = True
                                    traces.append(f"turnkey_only({ch.text}->{node.text})")
                # Adjacent "только под ключ" / "только с помещениями целиком"
                if tok.idx >= 1 and sent.tokens[tok.idx - 1].lemma in ("только", "исключительно"):
                    turnkey_only = True
                    traces.append(f"turnkey_only({sent.tokens[tok.idx-1].text} {tok.text})")

    # Check partial / separate work refusal: negated execution verb governing partial/small/separate work noun
    for tok in sent.tokens:
        if PARTIAL_MOD_RE.match(tok.lemma) or PARTIAL_MOD_RE.match(tok.text) or tok.lemma == "частичка":
            pred, _ = _find_governing_predicate(tok, sent)
            if pred is None:
                continue
            has_neg, neg_toks, price_cond, double_neg = _detect_predicate_negation(pred, sent)
            if has_neg and not price_cond and not double_neg:
                # Confirm the governed head or sibling is a work/order/room/service noun (or частичка)
                head_tok = sent.by_id.get(tok.head_id)
                is_work_np = (
                    tok.lemma == "частичка"
                    or (head_tok is not None and (SERVICE_WORK_NOUN_RE.match(head_tok.lemma) or SERVICE_WORK_NOUN_RE.match(head_tok.text)))
                    or any(SERVICE_WORK_NOUN_RE.match(c.lemma) for c in sent.children.get(pred.id, ()))
                )
                if is_work_np:
                    partial_refusal = True
                    neg_str = ",".join(t.text for t in neg_toks)
                    traces.append(f"partial_refusal({tok.text} -> {pred.text}[{neg_str}])")

    return turnkey_only, partial_refusal, traces


def _analyze_sentence_for_category(
    sent: ParsedSentence,
    cat_id: int,
    allow_anaphora: bool = False,
) -> dict[str, Any]:
    """Analyze a single sentence UD tree for mentions, actions, negations, actors, and temporal markers."""
    mentions = [t for t in sent.tokens if _token_matches_category(t, cat_id)]
    used_anaphora = False
    if not mentions and allow_anaphora:
        # Look for anaphoric service object ("эту работу", "эта задача", "такую работу", "такие заказы", "этим")
        # OR a standalone refusal predicate following a service-mention sentence ("Мы отказались.")
        for t in sent.tokens:
            if ANAPHORA_NOUN_RE.match(t.lemma) or ANAPHORA_NOUN_RE.match(t.text):
                if NON_SERVICE_NEG_OBJ_RE.match(t.lemma):
                    continue
                if t.rel in ("obj", "nsubj", "obl") or any(c.lemma in ("такой", "этот", "данный") for c in sent.children.get(t.id, ())):
                    mentions.append(t)
                    used_anaphora = True
                    break
        if not mentions:
            for t in sent.tokens:
                if t.lemma in REFUSAL_ACTION_LEMMAS or (t.lemma in ACTION_LEMMAS and any(_is_neg_particle(c, sent) for c in sent.children.get(t.id, ()))):
                    mentions.append(t)
                    used_anaphora = True
                    break

    if not mentions:
        return {"matched": False}

    clause_records = []
    min_dist_sa = -1
    min_dist_an = -1
    min_dist_sn = -1

    # Track clause-level root predicates in case of coordination ("Раньше не делали, теперь выполняем" or "попросили..., но получили отказ: нет")
    # Collect linked coordinated / parataxis predicates in the same sentence
    # Also handle anaphoric follow-up sentences where only refusal/offer is present
    seen_preds: set[str] = set()
    for mtok in mentions:
        pred, link_kind = _find_governing_predicate(mtok, sent)
        if pred is None:
            clause_records.append({
                "mention": mtok.text,
                "pred": None,
                "link_kind": "anaphora_cross_sent" if used_anaphora else "orphan_mention",
                "polarity": "unknown",
                "actor": "unknown",
                "temporal": "unknown",
                "not_only": False,
                "double_neg": False,
                "price_cond": False,
                "has_neg": False,
                "dist_sa": -1,
                "dist_an": -1,
                "dist_sn": -1,
            })
            continue

        candidate_preds = [pred]
        for ch in sent.children.get(pred.id, ()):
            if (ch.rel in ("conj", "parataxis") or (ch.rel == "iobj" and ch.pos == "VERB")) and _is_predicate_like(ch, sent):
                candidate_preds.append(ch)
        if pred.head_id in sent.by_id and pred.rel in ("xcomp", "ccomp", "advcl", "conj", "csubj"):
            parent_p = sent.by_id[pred.head_id]
            if _is_predicate_like(parent_p, sent):
                if parent_p.id not in [p.id for p in candidate_preds] and parent_p.lemma not in INQUIRY_LEMMAS | CLIENT_WISH_LEMMAS:
                    candidate_preds.append(parent_p)
                for ch in sent.children.get(parent_p.id, ()):
                    if (ch.rel in ("conj", "parataxis") or (ch.rel == "iobj" and ch.pos == "VERB")) and _is_predicate_like(ch, sent) and ch.id not in [p.id for p in candidate_preds]:
                        candidate_preds.append(ch)
        # Also include any other clause predicate in the SAME sentence that explicitly governs an anaphoric work/service NP
        # (handles "запрос ... устарел: ... не принимаем" and "не обещаем ... но по смете эту услугу оказываем")
        for st_tok in sent.tokens:
            if _is_predicate_like(st_tok, sent) and st_tok.id not in [p.id for p in candidate_preds]:
                if st_tok.lemma in REFUSAL_ACTION_LEMMAS or any(
                    (ANAPHORA_NOUN_RE.match(ch.lemma) or ANAPHORA_NOUN_RE.match(ch.text))
                    and ch.rel in ("obj", "obl", "nsubj")
                    for ch in sent.children.get(st_tok.id, ())
                ):
                    candidate_preds.append(st_tok)

        for cp in candidate_preds:
            if cp.id in seen_preds and len(candidate_preds) > 1:
                continue
            seen_preds.add(cp.id)

            if cp.lemma in INQUIRY_LEMMAS and len(candidate_preds) > 1:
                continue

            has_neg, neg_toks, price_cond, double_neg = _detect_predicate_negation(cp, sent)
            not_only = _find_not_only_in_clause(cp, sent)
            actor = _extract_clause_actor(cp, sent)
            temporal = _extract_clause_temporal(cp, sent, has_neg=has_neg, double_neg=double_neg)
            parent_head = sent.by_id.get(cp.head_id) if cp.head_id in sent.by_id else None
            is_under_inquiry = (
                cp.lemma in INQUIRY_LEMMAS
                or (
                    parent_head is not None
                    and parent_head.lemma in QUERY_META_LEMMAS
                    and cp.rel in ("nmod", "acl", "appos")
                )
                or (
                    cp.rel in ("xcomp", "ccomp")
                    and parent_head is not None
                    and (
                        parent_head.lemma in INQUIRY_LEMMAS
                        or (
                            parent_head.lemma in CLIENT_WISH_LEMMAS
                            and any(t.lemma in CLIENT_LEMMAS for t in sent.children.get(parent_head.id, ()) if t.rel == "nsubj")
                        )
                    )
                )
            )
            # "Клиент хочет/хотел заказать ..." when client is nsubj of client_wish -> not seller offer
            if cp.lemma in CLIENT_WISH_LEMMAS and any(t.lemma in CLIENT_LEMMAS for t in sent.children.get(cp.id, ()) if t.rel == "nsubj"):
                is_under_inquiry = True

            # Non-service object guard
            if has_neg and cp.id == pred.id:
                objs = [c for c in sent.children.get(cp.id, ()) if c.rel == "obj"]
                if objs and all(NON_SERVICE_NEG_OBJ_RE.match(o.lemma) for o in objs) and mtok.id not in [o.id for o in objs]:
                    has_neg = False
                    neg_toks = []

            d_sa = _shortest_ud_dist(sent, mtok.id, cp.id)
            d_an = min((_shortest_ud_dist(sent, cp.id, nt.id) for nt in neg_toks), default=-1) if neg_toks else -1
            d_sn = min((_shortest_ud_dist(sent, mtok.id, nt.id) for nt in neg_toks), default=-1) if neg_toks else -1

            if d_sa >= 0 and (min_dist_sa < 0 or d_sa < min_dist_sa):
                min_dist_sa = d_sa
            if d_an >= 0 and (min_dist_an < 0 or d_an < min_dist_an):
                min_dist_an = d_an
            if d_sn >= 0 and (min_dist_sn < 0 or d_sn < min_dist_sn):
                min_dist_sn = d_sn

            if is_under_inquiry and not has_neg and actor not in ("third_party", "client"):
                pol = "unknown"
            elif cp.id != pred.id and cp.lemma in INQUIRY_LEMMAS:
                pol = "unknown"
            elif actor in ("third_party", "client"):
                pol = "third_party"
            elif double_neg:
                pol = "double_neg_positive"
            elif price_cond and not has_neg:
                pol = "conditional"
            elif price_cond and has_neg:
                pol = "conditional_neg"
            elif has_neg:
                pol = "negated"
            elif temporal == "past":
                pol = "past_offer"
            elif not_only:
                pol = "offered"
            elif cp.lemma in ACTION_LEMMAS or actor == "seller_team" or link_kind == "direct_verb_obj":
                pol = "offered"
            else:
                pol = "unknown"

            clause_records.append({
                "mention": mtok.text,
                "pred": f"{cp.text}({cp.lemma})",
                "link_kind": "anaphora_cross_sent" if used_anaphora else link_kind,
                "polarity": pol,
                "actor": actor,
                "temporal": temporal,
                "not_only": not_only,
                "double_neg": double_neg,
                "price_cond": price_cond,
                "has_neg": has_neg,
                "neg_tokens": [t.text for t in neg_toks],
                "dist_sa": d_sa,
                "dist_an": d_an,
                "dist_sn": d_sn,
            })

    return {
        "matched": True,
        "used_anaphora": used_anaphora,
        "clauses": clause_records,
        "min_dist_sa": min_dist_sa,
        "min_dist_an": min_dist_an,
        "min_dist_sn": min_dist_sn,
    }


def extract_chunk_syntax_signals(analyzer: SyntaxAnalyzer, chunk_text: str, cat_id: int) -> dict[str, Any]:
    """Extract structured syntactic signals for one chunk and target category."""
    sents = analyzer.parse_text(chunk_text)
    all_clauses: list[dict[str, Any]] = []
    min_sa = -1
    min_an = -1
    min_sn = -1
    prev_sent_matched = False

    for sent in sents:
        res = _analyze_sentence_for_category(sent, cat_id, allow_anaphora=prev_sent_matched)
        if res["matched"]:
            # If sentence 1 was matched ("В старых тегах осталось: монтаж...") and sentence 2 uses anaphora ("Это не наше..., такую работу не берём"), keep tracking!
            prev_sent_matched = True
            all_clauses.extend(res["clauses"])
            if res["min_dist_sa"] >= 0 and (min_sa < 0 or res["min_dist_sa"] < min_sa):
                min_sa = res["min_dist_sa"]
            if res["min_dist_an"] >= 0 and (min_an < 0 or res["min_dist_an"] < min_an):
                min_an = res["min_dist_an"]
            if res["min_dist_sn"] >= 0 and (min_sn < 0 or res["min_dist_sn"] < min_sn):
                min_sn = res["min_dist_sn"]
        else:
            prev_sent_matched = False

    return {
        "matched": bool(all_clauses),
        "clauses": all_clauses,
        "min_dist_sa": min_sa,
        "min_dist_an": min_an,
        "min_dist_sn": min_sn,
    }


def detect_chunk_global_restriction(analyzer: SyntaxAnalyzer, chunk_text: str) -> tuple[int, int, list[str]]:
    """Detect turnkey-only / partial-refusal restrictions in a single chunk."""
    if not GLOBAL_CANDIDATE_RE.search(chunk_text):
        return 0, 0, []
    tk_flag = 0
    pr_flag = 0
    traces: list[str] = []
    for sent in analyzer.parse_text(chunk_text):
        tk, pr, tr = _detect_global_restrictions_in_sentence(sent)
        if tk:
            tk_flag = 1
        if pr:
            pr_flag = 1
        traces.extend(tr)
    return tk_flag, pr_flag, traces


def extract_ad_global_restrictions(
    analyzer: SyntaxAnalyzer,
    chunk_texts: list[str],
    precomputed_chunk_globals: dict[str, tuple[int, int, list[str]]] | None = None,
) -> dict[str, Any]:
    """Scan all chunks of an ad for syntactically verified global restrictions."""
    turnkey_cidxs: list[int] = []
    partial_ref_cidxs: list[int] = []
    traces: list[str] = []

    for c_pos, ch in enumerate(chunk_texts):
        if precomputed_chunk_globals is not None and ch in precomputed_chunk_globals:
            tk, pr, tr = precomputed_chunk_globals[ch]
        else:
            tk, pr, tr = detect_chunk_global_restriction(analyzer, ch)
        if tk:
            turnkey_cidxs.append(c_pos)
        if pr:
            partial_ref_cidxs.append(c_pos)
        traces.extend(tr)

    return {
        "global_turnkey_only": int(bool(turnkey_cidxs)),
        "global_partial_refusal": int(bool(partial_ref_cidxs)),
        "turnkey_chunk_positions": sorted(set(turnkey_cidxs)),
        "partial_refusal_chunk_positions": sorted(set(partial_ref_cidxs)),
        "traces": traces,
    }


def chunk_mentions_category(chunk_text: str, cat_id: int) -> int:
    """Fast lexical check whether chunk mentions target category (excluding weak modifiers)."""
    cat_pat = CATEGORY_PATTERNS[cat_id]
    for w in re.findall(r"[А-ЯЁа-яёA-Za-z]+", chunk_text):
        if cat_pat.match(w) and not WEAK_MOD_WORDS.match(w):
            return 1
    return 0


def extract_pair_features(
    analyzer: SyntaxAnalyzer | None,
    full_chunks: list[str],
    top3_positions: list[int],
    top3_sims: list[float],
    cat_id: int,
    ad_char_len: int,
    include_debug: bool = False,
    precomputed_chunk_signals: dict[tuple[str, int], dict[str, Any]] | None = None,
    precomputed_chunk_globals: dict[str, tuple[int, int, list[str]]] | None = None,
    precomputed_mentioned_anywhere: int | None = None,
) -> dict[str, Any]:
    """
    Build all structured syntactic + retrieval features for a single (ad, cat_id) pair.
    General across all categories 102..106. Uses unknown instead of False when links fail.
    """
    global_info = extract_ad_global_restrictions(
        analyzer, full_chunks, precomputed_chunk_globals=precomputed_chunk_globals
    )
    top3_set = set(top3_positions)

    # Check if category is mentioned anywhere in the full ad (even outside Top-3)
    if precomputed_mentioned_anywhere is not None:
        mentioned_anywhere = int(precomputed_mentioned_anywhere)
    else:
        mentioned_anywhere = 0
        for ch in full_chunks:
            if chunk_mentions_category(ch, cat_id):
                mentioned_anywhere = 1
                break

    top3_clauses: list[tuple[int, float, dict[str, Any]]] = []
    min_sa = -1
    min_an = -1
    min_sn = -1
    best_chunk_pol = "unknown"
    best_link_status = "not_found"
    top_offer_sim = 0.0
    top_neg_sim = 0.0

    for slot, (c_pos, sim_val) in enumerate(zip(top3_positions, top3_sims)):
        ch_text = full_chunks[c_pos]
        if precomputed_chunk_signals is not None and (ch_text, cat_id) in precomputed_chunk_signals:
            sig = precomputed_chunk_signals[(ch_text, cat_id)]
        else:
            assert analyzer is not None
            sig = extract_chunk_syntax_signals(analyzer, ch_text, cat_id)
        if not sig["matched"]:
            continue
        if sig["min_dist_sa"] >= 0 and (min_sa < 0 or sig["min_dist_sa"] < min_sa):
            min_sa = sig["min_dist_sa"]
        if sig["min_dist_an"] >= 0 and (min_an < 0 or sig["min_dist_an"] < min_an):
            min_an = sig["min_dist_an"]
        if sig["min_dist_sn"] >= 0 and (min_sn < 0 or sig["min_dist_sn"] < min_sn):
            min_sn = sig["min_dist_sn"]

        for cl in sig["clauses"]:
            top3_clauses.append((slot, sim_val, cl))

        # Slot-0 (highest similarity matched chunk) drives best_chunk_pol and best_link_status
        chunk_pols = [cl["polarity"] for cl in sig["clauses"]]
        if best_link_status == "not_found":
            links = [cl["link_kind"] for cl in sig["clauses"]]
            if "direct_verb_obj" in links:
                best_link_status = "direct_verb_obj"
            elif "prep_obl_or_nmod" in links:
                best_link_status = "prep_obl_or_nmod"
            elif "anaphora_cross_sent" in links:
                best_link_status = "anaphora_cross_sent"
            else:
                best_link_status = "orphan_mention"

        # Aggregate polarity of this matched chunk
        # Note temporal shift inside chunk ("past_neg + now_pos" -> offered; "past_pos + now_neg" -> negated)
        has_past_neg = any(cl["temporal"] == "past" and cl["polarity"] == "negated" for cl in sig["clauses"])
        has_now_pos = any(cl["temporal"] == "now" and cl["polarity"] == "offered" for cl in sig["clauses"])
        has_past_pos = any(cl["temporal"] == "past" and cl["polarity"] in ("offered", "past_offer") for cl in sig["clauses"])
        has_now_neg = any(cl["temporal"] in ("now", "stopped") and cl["polarity"] == "negated" for cl in sig["clauses"])

        if has_past_neg and has_now_pos:
            c_pol = "offered"
        elif has_past_pos and has_now_neg:
            c_pol = "negated"
        elif "double_neg_positive" in chunk_pols:
            c_pol = "double_neg_positive"
        elif "third_party" in chunk_pols and "seller_team" not in [cl["actor"] for cl in sig["clauses"] if cl["polarity"] == "offered"]:
            c_pol = "third_party"
        elif "negated" in chunk_pols and "offered" not in chunk_pols:
            c_pol = "negated"
        elif "conditional_neg" in chunk_pols and any(cl["polarity"] == "offered" for cl in sig["clauses"]):
            c_pol = "conditional"
        elif "conditional_neg" in chunk_pols:
            c_pol = "conditional"
        elif "offered" in chunk_pols:
            c_pol = "offered"
        else:
            c_pol = "unknown"

        if best_chunk_pol == "unknown":
            best_chunk_pol = c_pol
        if c_pol in ("offered", "double_neg_positive", "conditional") and sim_val > top_offer_sim:
            top_offer_sim = float(sim_val)
        if c_pol in ("negated", "third_party") and sim_val > top_neg_sim:
            top_neg_sim = float(sim_val)

    # Compute clause-level aggregates over Top-3
    has_action_linked = int(any(cl["pred"] is not None for _, _, cl in top3_clauses))
    has_neg_on_action = int(any(cl["has_neg"] and not cl["price_cond"] for _, _, cl in top3_clauses))
    has_not_only = int(any(cl["not_only"] for _, _, cl in top3_clauses))
    has_double_neg = int(any(cl["double_neg"] for _, _, cl in top3_clauses))
    has_price_cond_only = int(any(cl["price_cond"] for _, _, cl in top3_clauses) and not any(cl["has_neg"] and not cl["price_cond"] for _, _, cl in top3_clauses))

    # Effective offered / negated / third_party counts accounting for temporal shift and team contrast
    temporals = {cl["temporal"] for _, _, cl in top3_clauses if cl["temporal"] != "unknown"}
    has_past_neg = any(cl["temporal"] == "past" and cl["polarity"] == "negated" for _, _, cl in top3_clauses)
    has_now_pos = any(cl["temporal"] == "now" and cl["polarity"] == "offered" for _, _, cl in top3_clauses)
    has_past_pos = any(cl["temporal"] == "past" and cl["polarity"] in ("offered", "past_offer") for _, _, cl in top3_clauses)
    has_now_neg = any(cl["temporal"] in ("now", "stopped") and cl["polarity"] == "negated" for _, _, cl in top3_clauses)

    if has_past_neg and has_now_pos:
        temporal_state = "past_neg_to_now_pos"
    elif has_past_pos and has_now_neg:
        temporal_state = "past_pos_to_now_neg"
    elif "stopped" in temporals:
        temporal_state = "stopped"
    elif "now" in temporals:
        temporal_state = "now_only"
    elif "past" in temporals:
        temporal_state = "past_only"
    else:
        temporal_state = "unknown"

    actors = {cl["actor"] for _, _, cl in top3_clauses if cl["actor"] != "unknown"}
    has_third_party = int(bool(actors & {"third_party", "client"}))
    has_seller_actor = int("seller_team" in actors)
    if len(actors) == 0:
        dominant_subject = "unknown"
    elif actors == {"seller_team"}:
        dominant_subject = "seller_team"
    elif actors <= {"third_party", "client"}:
        dominant_subject = "third_party" if "third_party" in actors else "client"
    else:
        dominant_subject = "mixed"

    n_offered = sum(
        1 for _, _, cl in top3_clauses
        if cl["polarity"] in ("offered", "double_neg_positive", "conditional")
    )
    n_negated = sum(
        1 for _, _, cl in top3_clauses
        if cl["polarity"] in ("negated", "conditional_neg") and not cl["price_cond"]
    )
    n_third_party = sum(1 for _, _, cl in top3_clauses if cl["polarity"] == "third_party")

    # Discount past negated clause when resumed now, and vice versa
    if temporal_state == "past_neg_to_now_pos":
        n_negated = max(0, n_negated - 1)
    if temporal_state == "past_pos_to_now_neg":
        n_offered = max(0, n_offered - 1)

    has_affirmative_offer = int(n_offered > 0)
    # Conflicting claims: both positive offer AND negative/third-party/global-partial refusal present
    restr_positions = set(global_info["turnkey_chunk_positions"]) | set(global_info["partial_refusal_chunk_positions"])
    global_outside_top3 = int(bool(restr_positions - top3_set))
    has_conflicting_claims = int(
        (n_offered > 0 and (n_negated > 0 or n_third_party > 0))
        or temporal_state in ("past_neg_to_now_pos", "past_pos_to_now_neg")
        or (n_offered > 0 and global_info["global_partial_refusal"] == 1)
    )

    s0 = float(top3_sims[0]) if len(top3_sims) >= 1 else 0.0
    s1 = float(top3_sims[1]) if len(top3_sims) >= 2 else 0.0
    s2 = float(top3_sims[2]) if len(top3_sims) >= 3 else 0.0

    feat = {
        # Metadata / Qwen retrieval features (shared with Variant B control)
        "cat_id": str(cat_id),
        "ad_char_len": int(ad_char_len),
        "n_chunks": int(len(full_chunks)),
        "n_padded_slots": int(max(0, 3 - len(top3_positions))),
        "sim_slot0": round(s0, 6),
        "sim_slot1": round(s1, 6),
        "sim_slot2": round(s2, 6),
        "sim_max": round(max(top3_sims) if top3_sims else 0.0, 6),
        "sim_mean": round(float(sum(top3_sims) / len(top3_sims)) if top3_sims else 0.0, 6),
        "sim_margin_01": round(s0 - s1, 6),
        # Categorical syntax features (unknown preserved explicitly)
        "best_chunk_polarity": best_chunk_pol,
        "dominant_subject": dominant_subject,
        "temporal_state": temporal_state,
        "service_link_status": best_link_status,
        # Binary syntax features
        "syn_service_mentioned_top3": int(bool(top3_clauses)),
        "syn_service_mentioned_ad": int(mentioned_anywhere),
        "syn_action_linked": has_action_linked,
        "syn_has_neg_on_action": has_neg_on_action,
        "syn_has_not_only": has_not_only,
        "syn_double_neg_cancel": has_double_neg,
        "syn_affirmative_offer": has_affirmative_offer,
        "syn_third_party_actor": has_third_party,
        "syn_seller_actor": has_seller_actor,
        "syn_temporal_stopped": int(temporal_state in ("stopped", "past_pos_to_now_neg")),
        "syn_temporal_resumed": int(temporal_state == "past_neg_to_now_pos"),
        "syn_price_condition_only": has_price_cond_only,
        "syn_global_turnkey_only": int(global_info["global_turnkey_only"]),
        "syn_global_partial_refusal": int(global_info["global_partial_refusal"]),
        "syn_global_outside_top3": global_outside_top3,
        "syn_has_conflicting_claims": has_conflicting_claims,
        # Numeric syntax / UD graph features (-1 sentinel for missing link/distance)
        "syn_min_dist_service_action": int(min_sa),
        "syn_min_dist_action_neg": int(min_an),
        "syn_min_dist_service_neg": int(min_sn),
        "syn_n_offered_clauses_top3": int(n_offered),
        "syn_n_negated_clauses_top3": int(n_negated),
        "syn_n_third_party_clauses_top3": int(n_third_party),
        "syn_net_polarity_score": int(n_offered - n_negated - n_third_party),
        "syn_top_offer_sim": round(top_offer_sim, 6),
        "syn_top_neg_sim": round(top_neg_sim, 6),
    }

    if include_debug:
        feat["_debug_clauses"] = [cl for _, _, cl in top3_clauses]
        feat["_debug_global_traces"] = global_info["traces"]
    return feat


SYNTAX_CAT_FEATURES = [
    "cat_id",
    "best_chunk_polarity",
    "dominant_subject",
    "temporal_state",
    "service_link_status",
]

NON_SYNTAX_META_FEATURES = [
    "cat_id",
    "ad_char_len",
    "n_chunks",
    "n_padded_slots",
    "sim_slot0",
    "sim_slot1",
    "sim_slot2",
    "sim_max",
    "sim_mean",
    "sim_margin_01",
]

SYNTAX_ONLY_FEATURES = [
    "best_chunk_polarity",
    "dominant_subject",
    "temporal_state",
    "service_link_status",
    "syn_service_mentioned_top3",
    "syn_service_mentioned_ad",
    "syn_action_linked",
    "syn_has_neg_on_action",
    "syn_has_not_only",
    "syn_double_neg_cancel",
    "syn_affirmative_offer",
    "syn_third_party_actor",
    "syn_seller_actor",
    "syn_temporal_stopped",
    "syn_temporal_resumed",
    "syn_price_condition_only",
    "syn_global_turnkey_only",
    "syn_global_partial_refusal",
    "syn_global_outside_top3",
    "syn_has_conflicting_claims",
    "syn_min_dist_service_action",
    "syn_min_dist_action_neg",
    "syn_min_dist_service_neg",
    "syn_n_offered_clauses_top3",
    "syn_n_negated_clauses_top3",
    "syn_n_third_party_clauses_top3",
    "syn_net_polarity_score",
    "syn_top_offer_sim",
    "syn_top_neg_sim",
]
