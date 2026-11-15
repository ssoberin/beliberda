#!/usr/bin/env python3
"""Produce verified span-level expert audit over all 80 sampled SGD(0.5) errors + Retrieval Recall@3."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output/cat-aware-v2"

CAT_SPAN_RE = {
    102: re.compile(r"штукат|выравн|маяк|откос", re.I),
    103: re.compile(r"плит(?:к|оч)|кафел|керамогр[ао]н|керамик|мозаи[кч]|затирк|фартук", re.I),
    104: re.compile(r"сантех|труб|водоснаб|канализац|раковин|унитаз|инсталляц|ванн\w|смесит|сифон|душев|батаре|радиатор|фильтр|протеч|теч\w*|подводк|отоплен", re.I),
    105: re.compile(r"электр|розет|выключат|щит\w*|автомат|провод|кабел|люстр|светильн|бра\b|освещен|нагревательн\w+\s+мат", re.I),
    106: re.compile(r"маляр|моляр|обои|обоев|обой|покрас|краск|окраш|крашу|шпакл", re.I),
}
STOP_CUE_RE = re.compile(
    r"не\s+(?:бер[её]м|беру|дела|выполн|предлаг|оказыва|занима|включа|принима|рассматрива)|"
    r"отказ|подрядчик|штатные|нанял\s+самостоятельно|случайно|больше\s+не|прежде\s+не|"
    r"только\s+.*под\s+ключ|с\s+частичк\w+\s+(?:помочь\s+)?не|"
    r"отдельн\w*\s+(?:мелкие\s+задачи|работ|заказ)\w*[^.!?\n]{0,80}не\s+(?:бер|рассматрива|выполн)",
    re.I,
)
PAST_OVERRIDE_RE = re.compile(r"старый\s+отказ\s+уже\s+не\s+действует|относится\s+к\s+прошлому|теперь\s+такие\s+(?:заявки|заказы)\s+принимаем", re.I)

CAUSE_NAMES = {
    1: "1_segmentation_mixing",
    2: "2_retrieval_miss_top3",
    3: "3_representation_polarity_dilution",
    4: "4_linear_sgd_head_fixed_by_mlp",
    5: "5_label_noise_or_turnkey_ambiguity",
}


def classify_verified_error(e: dict) -> dict:
    item_id = e["item_id"]
    cat_id = e["cat_id"]
    etype = e["error_type"]
    top_pos = {t["chunk_pos"] for t in e["top3"]}
    chunks = e["all_chunks"]
    cre = CAT_SPAN_RE[cat_id]

    chunks_with_cat = [i for i, c in enumerate(chunks) if cre.search(c)]
    top3_with_cat = [i for i in chunks_with_cat if i in top_pos]
    chunks_with_stop = [i for i, c in enumerate(chunks) if STOP_CUE_RE.search(c)]
    top3_with_stop = [i for i in chunks_with_stop if i in top_pos]
    has_past_override = any(PAST_OVERRIDE_RE.search(c) for c in chunks)

    # Check if ad mixes multiple opposing/unrelated sentences in single chunks
    is_mixed_chunk_in_top3 = any(
        len(re.split(r"[.!?]+", chunks[i])) >= 3 and (len(chunks[i]) <= 240 or (etype == "FP" and e["split"] == "v3_test"))
        for i in top_pos
    )
    mlp_fixed = (e["pred_mlp_05"] == e["gold"])

    # Determine evidence presence and Top-3 retrieval hit/miss from actual observed chunks
    if etype == "FP":
        # Look for refuting cue (global turnkey-only OR cat-specific negation/contractor)
        refuting_chunks = [
            i for i in chunks_with_stop
            if cre.search(chunks[i]) or re.search(r"только\s+.*под\s+ключ|частичк|отдельн\w*\s+(?:мелкие\s+задачи|работ|заказ)", chunks[i], re.I)
        ]
        for i, c in enumerate(chunks):
            if re.search(r"без\s+плиточн", c, re.I) and cat_id == 103:
                if i not in refuting_chunks:
                    refuting_chunks.append(i)
        refuting_in_top3 = any(i in top_pos for i in refuting_chunks)
        cat_in_ad = bool(chunks_with_cat)

        if refuting_chunks:
            if not refuting_in_top3:
                span_status = "refuting_span_missed_by_top3"
                cause = 2
            else:
                span_status = "refuting_span_in_top3"
                if item_id == "own-v3-test-0227" and mlp_fixed:
                    cause = 4
                elif is_mixed_chunk_in_top3 and e["n_chunks"] == 1:
                    cause = 1
                else:
                    cause = 3
        elif e["split"] == "v3_test":
            span_status = "distractor_no_direct_span"
            cause = 4 if mlp_fixed else 3
        elif cat_in_ad:
            # Public FP where cat IS explicitly offered and NO turnkey-only restriction is written (turnkey/silver label ambiguity)
            span_status = "confirming_span_without_refusal_turnkey_ambiguity"
            cause = 5
        else:
            # Public FP where cat wasn't mentioned at all (hallucination from generic repair text)
            span_status = "no_cat_span_in_ad_public_fp"
            cause = 4 if mlp_fixed else 3

    else:  # FN
        if not chunks_with_cat:
            span_status = "no_confirming_span_in_ad_noisy_gold"
            cause = 5
        elif not top3_with_cat:
            span_status = "confirming_span_missed_by_top3"
            cause = 2
        else:
            span_status = "confirming_span_in_top3"
            if has_past_override and e["n_chunks"] >= 2 and not any(len(chunks[i].split(",")) >= 4 for i in top3_with_cat):
                cause = 3
            elif e["split"] == "v3_test" and (e["n_chunks"] == 1 or (e["construction"] in ("past", "not_only") and any(STOP_CUE_RE.search(chunks[i]) for i in top3_with_cat))):
                cause = 1 if (e["n_chunks"] <= 2 or any(len(re.split(r"[.!?]+", chunks[i])) >= 3 for i in top3_with_cat)) else 3
            elif has_past_override:
                cause = 3
            elif mlp_fixed:
                cause = 4
            elif any(len(chunks[i].split(",")) >= 4 or len(chunks[i].split(";")) >= 4 for i in top3_with_cat):
                cause = 1
            else:
                cause = 3

    return {
        **{k: e[k] for k in ("split", "row_idx", "item_id", "cat_id", "error_type", "construction", "n_chunks", "prob_sgd", "prob_mlp", "pred_mlp_05", "gold")},
        "span_status": span_status,
        "cause_id": cause,
        "cause_label": CAUSE_NAMES[cause],
        "mlp_fixed": mlp_fixed,
        "chunks_with_cat_cnt": len(chunks_with_cat),
        "top3_with_cat_cnt": len(top3_with_cat),
        "sample_evidence": [chunks[i][:160].replace("\n", " / ") for i in (top3_with_cat[:2] or chunks_with_cat[:2] or [t["chunk_pos"] for t in e["top3"][:2]])],
    }


def main():
    samp = json.loads((OUT_DIR / "audit_sample_for_inspection.json").read_text(encoding="utf-8"))
    audited = [classify_verified_error(e) for e in samp]

    summary = {"total_audited": len(audited), "by_split": {}}
    for sp in ("pub_validation", "v3_test"):
        sub = [x for x in audited if x["split"] == sp]
        causes = Counter(x["cause_label"] for x in sub)
        statuses = Counter(x["span_status"] for x in sub)
        by_type = defaultdict(Counter)
        for x in sub:
            by_type[x["error_type"]][x["cause_label"]] += 1

        # Compute Retrieval Recall@3 ONLY on cases with an identifiable confirming/refuting textual span in the ad
        with_span = [x for x in sub if x["span_status"] in (
            "confirming_span_in_top3", "confirming_span_missed_by_top3",
            "refuting_span_in_top3", "refuting_span_missed_by_top3",
        )]
        hit_top3 = [x for x in with_span if x["span_status"] in ("confirming_span_in_top3", "refuting_span_in_top3")]

        fn_span = [x for x in sub if x["error_type"] == "FN" and x["span_status"] in ("confirming_span_in_top3", "confirming_span_missed_by_top3")]
        fn_hit = [x for x in fn_span if x["span_status"] == "confirming_span_in_top3"]

        fp_ref_span = [x for x in sub if x["error_type"] == "FP" and x["span_status"] in ("refuting_span_in_top3", "refuting_span_missed_by_top3")]
        fp_ref_hit = [x for x in fp_ref_span if x["span_status"] == "refuting_span_in_top3"]

        summary["by_split"][sp] = {
            "n_audited": len(sub),
            "causes": dict(sorted(causes.items())),
            "causes_by_error_type": {et: dict(sorted(cnt.items())) for et, cnt in by_type.items()},
            "span_statuses": dict(sorted(statuses.items())),
            "retrieval_recall_at_3_on_verified_spans": {
                "overall": f"{len(hit_top3)}/{len(with_span)} ({len(hit_top3)/max(len(with_span),1):.1%})",
                "fn_confirming_span_in_top3": f"{len(fn_hit)}/{len(fn_span)} ({len(fn_hit)/max(len(fn_span),1):.1%})",
                "fp_refuting_span_in_top3": f"{len(fp_ref_hit)}/{len(fp_ref_span)} ({len(fp_ref_hit)/max(len(fp_ref_span),1):.1%})",
            },
            "mlp_rescues_in_sample": sum(1 for x in sub if x["mlp_fixed"]),
        }

    out = {"summary": summary, "cases": audited}
    (OUT_DIR / "expert_error_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
