"""Comprehensive collision, evaluation, overfitting, and error-case analysis for n-gram expansion."""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "output/v2-complexity-audit"
sys.path.insert(0, str(AUDIT_DIR))
import analyze as complexity_audit  # noqa: E402

IDS = (102, 103, 104, 105, 106)
CONFIG_NAMES = [
    "ref_46k",
    "A_1_3_uncapped",
    "B_1_4_uncapped",
    "C_1_5_uncapped",
    "D_1_6_uncapped",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    if len(gold) == 0:
        return {"rows": 0}
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    return {
        "rows": int(len(gold)),
        "micro_f1": float(f1_score(gold, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(gold, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(gold, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(gold, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(gold, pred, average="macro", zero_division=0)),
        "exact_set_accuracy": float(accuracy_score(gold, pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "empty_predictions": int((pred.sum(axis=1) == 0).sum()),
        "empty_on_positive_rows": int(((pred.sum(axis=1) == 0) & (gold.sum(axis=1) > 0)).sum()),
        "positive_rows": int((gold.sum(axis=1) > 0).sum()),
        "gold_empty_rows": int((gold.sum(axis=1) == 0).sum()),
        "f1_by_category": dict(zip(map(str, IDS), map(float, f1_score(gold, pred, average=None, zero_division=0)))),
    }


def paired_bootstrap_micro_f1(gold: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(gold), size=(n_boot, len(gold)))
    scores = []
    for pred in (pred_a, pred_b):
        per_row = np.column_stack([
            ((gold == 1) & (pred == 1)).sum(axis=1),
            ((gold == 0) & (pred == 1)).sum(axis=1),
            ((gold == 1) & (pred == 0)).sum(axis=1),
        ])
        s = per_row[idx].sum(axis=1)
        scores.append(2 * s[:, 0] / np.maximum(1, 2 * s[:, 0] + s[:, 1] + s[:, 2]))
    diff = scores[1] - scores[0]
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return {
        "delta_micro_f1": float(
            f1_score(gold, pred_b, average="micro", zero_division=0)
            - f1_score(gold, pred_a, average="micro", zero_division=0)
        ),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def vector_fingerprint(csr_row) -> str:
    return hashlib.sha256(csr_row.indices.tobytes() + csr_row.data.tobytes()).hexdigest()


def analyze_collisions_for_split(rows: list[dict], gold: np.ndarray, vectorizer, word_vec, char_vec) -> dict:
    texts = [r["description"][:6000] for r in rows]
    mat = vectorizer.transform(texts).tocsr()
    w_mat = word_vec.transform(texts).tocsr()
    c_mat = char_vec.transform(texts).tocsr()

    vec_groups = collections.defaultdict(list)
    src_vec_groups = collections.defaultdict(list)
    word_only_groups = collections.defaultdict(list)
    char_only_groups = collections.defaultdict(list)

    fingerprints = []
    for i, r in enumerate(rows):
        a, b = mat.indptr[i : i + 2]
        fp = hashlib.sha256(mat.indices[a:b].tobytes() + mat.data[a:b].tobytes()).hexdigest()
        fingerprints.append(fp)
        vec_groups[fp].append(i)
        src_vec_groups[(r.get("mcId", 101), fp)].append(i)

        wa, wb = w_mat.indptr[i : i + 2]
        wfp = hashlib.sha256(w_mat.indices[wa:wb].tobytes() + w_mat.data[wa:wb].tobytes()).hexdigest()
        word_only_groups[wfp].append(i)

        ca, cb = c_mat.indptr[i : i + 2]
        cfp = hashlib.sha256(c_mat.indices[ca:cb].tobytes() + c_mat.data[ca:cb].tobytes()).hexdigest()
        char_only_groups[cfp].append(i)

    # 1. Identical vector groups regardless of source/gold
    any_vec_coll = [idxs for idxs in vec_groups.values() if len(idxs) > 1]
    any_src_vec_coll = [idxs for idxs in src_vec_groups.values() if len(idxs) > 1]

    # 2. Conflicting groups (different gold evaluation_labels)
    conflicting_vec_eval = [idxs for idxs in any_vec_coll if len({tuple(gold[i]) for i in idxs}) > 1]
    conflicting_vec_offered = [
        idxs for idxs in any_vec_coll if len({tuple(sorted(rows[i].get("labels", []))) for i in idxs}) > 1
    ]
    conflicting_src_vec = [idxs for idxs in any_src_vec_coll if len({tuple(gold[i]) for i in idxs}) > 1]

    unavoidable_label_errors = sum(
        int(np.minimum(gold[idxs].sum(axis=0), len(idxs) - gold[idxs].sum(axis=0)).sum())
        for idxs in conflicting_src_vec
    )
    unavoidable_exact_errors = sum(
        len(idxs) - collections.Counter(tuple(gold[i]) for i in idxs).most_common(1)[0][1]
        for idxs in conflicting_src_vec
    )

    return {
        "fingerprints": fingerprints,
        "src_vec_groups": src_vec_groups,
        "conflicting_src_vec": conflicting_src_vec,
        "summary": {
            "total_rows": len(rows),
            "unique_texts": len({r["description"][:6000] for r in rows}),
            "unique_combined_vectors": len(vec_groups),
            "unique_word_vectors": len(word_only_groups),
            "unique_char_vectors": len(char_only_groups),
            "vector_collision_groups_any": len(any_vec_coll),
            "vector_collision_rows_any": sum(map(len, any_vec_coll)),
            "src_vector_collision_groups_any": len(any_src_vec_coll),
            "src_vector_collision_rows_any": sum(map(len, any_src_vec_coll)),
            "conflicting_vec_groups_eval_labels": len(conflicting_vec_eval),
            "conflicting_vec_rows_eval_labels": sum(map(len, conflicting_vec_eval)),
            "conflicting_vec_groups_offered_labels": len(conflicting_vec_offered),
            "conflicting_vec_rows_offered_labels": sum(map(len, conflicting_vec_offered)),
            "conflicting_src_vec_groups": len(conflicting_src_vec),
            "conflicting_src_vec_rows": sum(map(len, conflicting_src_vec)),
            "conflicting_src_vec_rows_by_type": dict(
                collections.Counter(rows[i].get("type", "public") for idxs in conflicting_src_vec for i in idxs)
            ),
            "conflicting_src_vec_groups_by_type": dict(
                collections.Counter(
                    "|".join(sorted({rows[i].get("type", "public") for i in idxs})) for idxs in conflicting_src_vec
                )
            ),
            "minimum_unavoidable_label_errors": unavoidable_label_errors,
            "minimum_unavoidable_exact_set_errors": unavoidable_exact_errors,
        },
    }


def diagnose_why_pair_collides(text_a: str, text_b: str, word_vec, word_ngram_max: int) -> dict:
    """Explain why two texts still collide or which n-grams distinguish them."""
    analyze_w = word_vec.build_analyzer()
    toks_a = set(analyze_w(text_a))
    toks_b = set(analyze_w(text_b))
    vocab = word_vec.vocabulary_
    in_vocab_a = {t for t in toks_a if t in vocab}
    in_vocab_b = {t for t in toks_b if t in vocab}
    diff_in_vocab_a = sorted(in_vocab_a - in_vocab_b, key=lambda s: (s.count(" ") + 1, s))
    diff_in_vocab_b = sorted(in_vocab_b - in_vocab_a, key=lambda s: (s.count(" ") + 1, s))
    diff_oov_a = sorted((toks_a - toks_b) - vocab.keys(), key=lambda s: (s.count(" ") + 1, s))
    diff_oov_b = sorted((toks_b - toks_a) - vocab.keys(), key=lambda s: (s.count(" ") + 1, s))

    # Line-by-line token span analysis
    lines_a = [ln.strip() for ln in text_a.splitlines() if ln.strip()]
    lines_b = [ln.strip() for ln in text_b.splitlines() if ln.strip()]
    word_re = re.compile(r"(?u)\b\w\w+\b")
    differing_lines = []
    for la, lb in zip(lines_a, lines_b):
        if la != lb:
            wa = word_re.findall(la.lower())
            wb = word_re.findall(lb.lower())
            differing_lines.append({
                "line_a": la,
                "line_b": lb,
                "word_tokens_a": len(wa),
                "word_tokens_b": len(wb),
            })
    return {
        "in_vocab_diff_a": diff_in_vocab_a,
        "in_vocab_diff_b": diff_in_vocab_b,
        "oov_diff_a_count": len(diff_oov_a),
        "oov_diff_b_count": len(diff_oov_b),
        "oov_diff_a_sample": diff_oov_a[:12],
        "oov_diff_b_sample": diff_oov_b[:12],
        "differing_lines": differing_lines,
        "word_ngram_max": word_ngram_max,
    }


def main() -> None:
    exp_dir = ROOT / "output/ngram-expansion-experiment"
    splits_dir = exp_dir / "splits"
    runs_dir = exp_dir / "runs"

    train_rows = read_jsonl(splits_dir / "train.jsonl")
    val_rows = read_jsonl(splits_dir / "validation.jsonl")
    challenge_rows = read_jsonl(splits_dir / "challenge.jsonl")

    public_mask = np.asarray([r["dataset"] == "public" for r in val_rows])
    v2_mask = np.asarray([r["dataset"] == "v2" for r in val_rows])
    v2_raw_meta = {r["item_id"]: r for r in read_jsonl(ROOT / "datasets/hard_dataset/v2/test.jsonl")}
    v2_rows = []
    for r in val_rows:
        if r["dataset"] == "v2":
            meta = v2_raw_meta[r["item_id"]]
            merged = dict(r)
            for k in ("type", "group_id", "augmentation", "negation_depth"):
                if k in meta:
                    merged[k] = meta[k]
            v2_rows.append(merged)
    v2_texts = [r["description"][:6000] for r in v2_rows]
    v2_realism = np.asarray([complexity_audit.realism(t)[0] for t in v2_texts])
    v2_types = np.asarray([r.get("type", "unknown") for r in v2_rows])

    train_public_mask = np.asarray([r["dataset"] == "public" for r in train_rows])
    train_v2_mask = np.asarray([r["dataset"] == "v2" for r in train_rows])

    # Load baseline collision_proof.json for cross-verification
    baseline_collision_proof = json.loads((AUDIT_DIR / "collision_proof.json").read_text(encoding="utf-8"))

    models_data = {}
    for name in CONFIG_NAMES:
        run_dir = runs_dir / name
        meas = json.loads((run_dir / "measurement.json").read_text(encoding="utf-8"))
        bundle = joblib.load(run_dir / "tfidf_baseline.joblib")
        npz = np.load(run_dir / "predictions.npz")
        models_data[name] = {
            "measurement": meas,
            "pipeline": bundle["pipeline"],
            "val_gold": npz["val_gold"],
            "val_pred": npz["val_pred"],
            "val_prob": npz["val_prob"],
            "chal_gold": npz["challenge_gold"],
            "chal_pred": npz["challenge_pred"],
            "chal_prob": npz["challenge_prob"],
            "train_eval_gold": npz["train_eval_gold"],
            "train_eval_pred": npz["train_eval_pred"],
            "train_prob": npz["train_prob"],
        }

    # --- 1. Collision Analysis on fixed v2_test (3016) & public_validation (729) ---
    v2_gold = models_data["ref_46k"]["val_gold"][v2_mask]
    pub_gold = models_data["ref_46k"]["val_gold"][public_mask]

    collisions_by_model = {}
    pub_collisions_by_model = {}
    for name in CONFIG_NAMES:
        pipe = models_data[name]["pipeline"]
        vec = pipe.named_steps["vectorizer"]
        tr = dict(vec.transformer_list)
        collisions_by_model[name] = analyze_collisions_for_split(v2_rows, v2_gold, vec, tr["word"], tr["char"])
        pub_rows = [r for r in val_rows if r["dataset"] == "public"]
        pub_collisions_by_model[name] = analyze_collisions_for_split(pub_rows, pub_gold, vec, tr["word"], tr["char"])

    # Verify ref_46k matches baseline_collision_proof (231 groups, 642 rows)
    ref_coll_summary = collisions_by_model["ref_46k"]["summary"]
    assert ref_coll_summary["conflicting_src_vec_groups"] == baseline_collision_proof["conflicting_groups"], ref_coll_summary
    assert ref_coll_summary["conflicting_src_vec_rows"] == baseline_collision_proof["rows"], ref_coll_summary

    # Track the exact 231 baseline conflicting groups across ref_46k -> A -> B -> C -> D
    baseline_231_groups = collisions_by_model["ref_46k"]["conflicting_src_vec"]
    baseline_642_indices = sorted(i for idxs in baseline_231_groups for i in idxs)
    baseline_group_tracking = {}
    for name in CONFIG_NAMES:
        fps = collisions_by_model[name]["fingerprints"]
        resolved_no_conflict = 0
        fully_separated_vectors = 0
        still_conflicting = 0
        partially_separated_still_conflicting = 0
        unchanged_fully_colliding = 0
        resolved_by_type = collections.Counter()
        remaining_by_type = collections.Counter()
        resolved_group_ids = []
        remaining_group_ids = []

        for g_idx, idxs in enumerate(baseline_231_groups):
            typ = v2_rows[idxs[0]].get("type", "unknown")
            sub = collections.defaultdict(list)
            for i in idxs:
                sub[(v2_rows[i].get("mcId", 101), fps[i])].append(i)
            has_conflict = any(len({tuple(v2_gold[i]) for i in sub_idxs}) > 1 for sub_idxs in sub.values())
            if not has_conflict:
                resolved_no_conflict += 1
                resolved_by_type[typ] += 1
                resolved_group_ids.append(g_idx)
                if len(sub) == len(idxs):
                    fully_separated_vectors += 1
            else:
                still_conflicting += 1
                remaining_by_type[typ] += 1
                remaining_group_ids.append(g_idx)
                if len(sub) > 1:
                    partially_separated_still_conflicting += 1
                else:
                    unchanged_fully_colliding += 1

        baseline_group_tracking[name] = {
            "total_baseline_groups": len(baseline_231_groups),
            "resolved_no_conflict": resolved_no_conflict,
            "fully_separated_vectors": fully_separated_vectors,
            "still_conflicting": still_conflicting,
            "partially_separated_still_conflicting": partially_separated_still_conflicting,
            "unchanged_fully_colliding": unchanged_fully_colliding,
            "resolved_by_type": dict(resolved_by_type),
            "remaining_by_type": dict(remaining_by_type),
            "resolved_group_indices": resolved_group_ids,
            "remaining_group_indices": remaining_group_ids,
        }

    # Inspect WHICH new n-grams separated the newly resolved baseline groups at each step
    ngrams_resolving_groups = {}
    prev_resolved = set()
    for name in CONFIG_NAMES[1:]:
        curr_resolved = set(baseline_group_tracking[name]["resolved_group_indices"])
        newly_resolved = sorted(curr_resolved - prev_resolved)
        word_vec = dict(models_data[name]["pipeline"].named_steps["vectorizer"].transformer_list)["word"]
        ngram_max = models_data[name]["measurement"]["word_ngram_range"][1]
        examples = []
        ngram_len_counter = collections.Counter()
        new_ngram_examples_counter = collections.Counter()
        for g_idx in newly_resolved:
            idxs = baseline_231_groups[g_idx]
            # Pick a pair with different gold
            a = idxs[0]
            b = next(i for i in idxs if not np.array_equal(v2_gold[i], v2_gold[a]))
            diag = diagnose_why_pair_collides(v2_texts[a], v2_texts[b], word_vec, ngram_max)
            all_diff = diag["in_vocab_diff_a"] + diag["in_vocab_diff_b"]
            for tok in all_diff:
                ngram_len_counter[tok.count(" ") + 1] += 1
                if tok.count(" ") + 1 == ngram_max:
                    new_ngram_examples_counter[tok] += 1
            if len(examples) < 5:
                examples.append({
                    "group_idx": g_idx,
                    "type": v2_rows[a].get("type"),
                    "item_a": v2_rows[a]["item_id"],
                    "item_b": v2_rows[b]["item_id"],
                    "gold_a": [IDS[j] for j in np.flatnonzero(v2_gold[a])],
                    "gold_b": [IDS[j] for j in np.flatnonzero(v2_gold[b])],
                    "in_vocab_diff_a": diag["in_vocab_diff_a"][:10],
                    "in_vocab_diff_b": diag["in_vocab_diff_b"][:10],
                    "differing_lines": diag["differing_lines"],
                })
        ngrams_resolving_groups[name] = {
            "newly_resolved_count": len(newly_resolved),
            "distinguishing_ngram_lengths": {str(k): int(v) for k, v in sorted(ngram_len_counter.items())},
            "top_max_len_distinguishing_ngrams": new_ngram_examples_counter.most_common(15),
            "examples": examples,
        }
        prev_resolved = curr_resolved

    # Inspect remaining collisions in D_1_6_uncapped: why did they survive?
    d_word_vec = dict(models_data["D_1_6_uncapped"]["pipeline"].named_steps["vectorizer"].transformer_list)["word"]
    d_remaining = collisions_by_model["D_1_6_uncapped"]["conflicting_src_vec"]
    d_surviving_diagnostics = []
    surviving_reason_counts = collections.Counter()
    surviving_line_token_lengths = []
    for g_idx, idxs in enumerate(d_remaining):
        a = idxs[0]
        b = next(i for i in idxs if not np.array_equal(v2_gold[i], v2_gold[a]))
        diag = diagnose_why_pair_collides(v2_texts[a], v2_texts[b], d_word_vec, 6)
        has_oov_diff = (diag["oov_diff_a_count"] > 0 or diag["oov_diff_b_count"] > 0)
        max_line_tokens = max(
            [max(d["word_tokens_a"], d["word_tokens_b"]) for d in diag["differing_lines"]] or [0]
        )
        surviving_line_token_lengths.append(max_line_tokens)
        if has_oov_diff:
            surviving_reason_counts["has_oov_ngram_filtered_by_min_df_2"] += 1
        else:
            surviving_reason_counts["zero_ngram_diff_even_before_min_df_span_exceeds_6_words"] += 1
        if len(d_surviving_diagnostics) < 8:
            d_surviving_diagnostics.append({
                "type": v2_rows[a].get("type"),
                "source_mcId": v2_rows[a].get("mcId"),
                "group_size": len(idxs),
                "item_a": v2_rows[a]["item_id"],
                "item_b": v2_rows[b]["item_id"],
                "gold_a": [IDS[j] for j in np.flatnonzero(v2_gold[a])],
                "gold_b": [IDS[j] for j in np.flatnonzero(v2_gold[b])],
                "text_a": v2_texts[a],
                "text_b": v2_texts[b],
                "oov_diff_a_count": diag["oov_diff_a_count"],
                "oov_diff_b_count": diag["oov_diff_b_count"],
                "oov_diff_a_sample": diag["oov_diff_a_sample"][:6],
                "oov_diff_b_sample": diag["oov_diff_b_sample"][:6],
                "differing_lines": diag["differing_lines"],
            })

    # --- 2. Detailed Slice Evaluations ---
    slice_evaluations = {}
    for name in CONFIG_NAMES:
        d = models_data[name]
        v_gold = d["val_gold"]
        v_pred = d["val_pred"]
        v2_p = v_pred[v2_mask]
        v2_g = v_gold[v2_mask]

        status_conf_mask = (v2_types == "status_confirmation")
        status_neg_mask = (v2_types == "status_negated_denial")
        status_both_mask = status_conf_mask | status_neg_mask
        non_status_mask = ~status_both_mask

        # Metrics on the exact 642 rows of the baseline 231 collision groups
        coll_642_mask = np.zeros(len(v2_rows), dtype=bool)
        coll_642_mask[baseline_642_indices] = True

        # Other list/FAQ types in v2_test
        by_type = {}
        for typ in sorted(set(v2_types)):
            m = (v2_types == typ)
            by_type[typ] = compute_metrics(v2_g[m], v2_p[m])

        slice_evaluations[name] = {
            "public_validation": compute_metrics(v_gold[public_mask], v_pred[public_mask]),
            "v2_test": compute_metrics(v2_g, v2_p),
            "v2_realism": {
                grp: compute_metrics(v2_g[v2_realism == grp], v2_p[v2_realism == grp])
                for grp in ("realistic", "plausible_hard", "artificial")
            },
            "v2_problem_lists": {
                "status_confirmation": compute_metrics(v2_g[status_conf_mask], v2_p[status_conf_mask]),
                "status_negated_denial": compute_metrics(v2_g[status_neg_mask], v2_p[status_neg_mask]),
                "both_status_lists_1152": compute_metrics(v2_g[status_both_mask], v2_p[status_both_mask]),
                "v2_without_status_lists_1864": compute_metrics(v2_g[non_status_mask], v2_p[non_status_mask]),
                "baseline_231_collision_642_rows": compute_metrics(v2_g[coll_642_mask], v2_p[coll_642_mask]),
            },
            "v2_by_type": by_type,
            "challenge": compute_metrics(d["chal_gold"], d["chal_pred"]),
            "train_diagnostic": {
                "public_train": compute_metrics(d["train_eval_gold"][train_public_mask], d["train_eval_pred"][train_public_mask]),
                "v2_train": compute_metrics(d["train_eval_gold"][train_v2_mask], d["train_eval_pred"][train_v2_mask]),
            },
        }

    # --- 3. Paired Bootstrap & Flip Analysis vs ref_46k ---
    bootstrap_vs_ref = {}
    flips_vs_ref = {}
    ref_pred = models_data["ref_46k"]["val_pred"]
    ref_gold = models_data["ref_46k"]["val_gold"]
    for name in CONFIG_NAMES[1:]:
        curr_pred = models_data[name]["val_pred"]
        bootstrap_vs_ref[name] = {
            "public_validation": paired_bootstrap_micro_f1(ref_gold[public_mask], ref_pred[public_mask], curr_pred[public_mask]),
            "v2_test": paired_bootstrap_micro_f1(ref_gold[v2_mask], ref_pred[v2_mask], curr_pred[v2_mask]),
        }
        # Exact row-level and label-level flips
        pub_row_better = int((np.all(curr_pred[public_mask] == ref_gold[public_mask], axis=1) & ~np.all(ref_pred[public_mask] == ref_gold[public_mask], axis=1)).sum())
        pub_row_worse = int((~np.all(curr_pred[public_mask] == ref_gold[public_mask], axis=1) & np.all(ref_pred[public_mask] == ref_gold[public_mask], axis=1)).sum())
        v2_row_better = int((np.all(curr_pred[v2_mask] == ref_gold[v2_mask], axis=1) & ~np.all(ref_pred[v2_mask] == ref_gold[v2_mask], axis=1)).sum())
        v2_row_worse = int((~np.all(curr_pred[v2_mask] == ref_gold[v2_mask], axis=1) & np.all(ref_pred[v2_mask] == ref_gold[v2_mask], axis=1)).sum())
        flips_vs_ref[name] = {
            "public_exact_fixed": pub_row_better,
            "public_exact_broken": pub_row_worse,
            "v2_exact_fixed": v2_row_better,
            "v2_exact_broken": v2_row_worse,
        }

    # --- 4. Overfitting & Feature Weight Analysis ---
    weight_analysis = {}
    for name in CONFIG_NAMES:
        pipe = models_data[name]["pipeline"]
        vec = pipe.named_steps["vectorizer"]
        tr = dict(vec.transformer_list)
        word_vec = tr["word"]
        char_vec = tr["char"]
        word_dim = len(word_vec.vocabulary_)
        inv_word = {idx: tok for tok, idx in word_vec.vocabulary_.items()}
        inv_char = {idx + word_dim: tok for tok, idx in char_vec.vocabulary_.items()}

        clf = pipe.named_steps["classifier"]
        l2_sq_by_len = collections.defaultdict(float)
        nonzero_by_len = collections.defaultdict(int)
        top_long_ngrams = []
        for head_idx, est in enumerate(clf.estimators_):
            w = est.coef_[0]
            # word part
            w_word = w[:word_dim]
            for j, val in enumerate(w_word):
                nlen = inv_word[j].count(" ") + 1
                l2_sq_by_len[f"word_{nlen}"] += float(val * val)
                if abs(val) > 1e-6:
                    nonzero_by_len[f"word_{nlen}"] += 1
            # char part
            w_char = w[word_dim:]
            l2_sq_by_len["char_wb"] += float(np.dot(w_char, w_char))
            nonzero_by_len["char_wb"] += int((np.abs(w_char) > 1e-6).sum())

            if head_idx == 0:
                # collect sample top 4+ grams across heads later
                pass
        # Top absolute weight n-grams >=4 words across all 5 heads
        if models_data[name]["measurement"]["word_ngram_range"][1] >= 4:
            candidates = []
            for head_idx, est in enumerate(clf.estimators_):
                w = est.coef_[0][:word_dim]
                top_pos = np.argsort(w)[-10:][::-1]
                top_neg = np.argsort(w)[:10]
                for j in list(top_pos) + list(top_neg):
                    tok = inv_word[int(j)]
                    if tok.count(" ") + 1 >= 4:
                        candidates.append({
                            "mcId": IDS[head_idx],
                            "ngram": tok,
                            "len": tok.count(" ") + 1,
                            "weight": float(w[int(j)]),
                        })
            candidates.sort(key=lambda x: abs(x["weight"]), reverse=True)
            top_long_ngrams = candidates[:20]

        total_l2 = sum(l2_sq_by_len.values())
        weight_analysis[name] = {
            "l2_sq_share": {k: v / total_l2 for k, v in sorted(l2_sq_by_len.items())},
            "l2_sq_abs": {k: v for k, v in sorted(l2_sq_by_len.items())},
            "top_long_ngrams_sample": top_long_ngrams[:12],
        }

    # --- 5. Concrete Examples: Fixed vs Still Failing ---
    # Check v2_test and public_validation examples where D_1_6 (or B/C/D) fixed ref_46k's error
    ref_v2_pred = models_data["ref_46k"]["val_pred"][v2_mask]
    ref_v2_prob = models_data["ref_46k"]["val_prob"][v2_mask]
    d_v2_pred = models_data["D_1_6_uncapped"]["val_pred"][v2_mask]
    d_v2_prob = models_data["D_1_6_uncapped"]["val_prob"][v2_mask]
    ref_pub_pred = models_data["ref_46k"]["val_pred"][public_mask]
    ref_pub_prob = models_data["ref_46k"]["val_prob"][public_mask]
    d_pub_pred = models_data["D_1_6_uncapped"]["val_pred"][public_mask]
    d_pub_prob = models_data["D_1_6_uncapped"]["val_prob"][public_mask]
    pub_rows = [r for r in val_rows if r["dataset"] == "public"]

    # Helper to extract top n-grams (n>=4 or uncapped <=3) firing in a text with highest signed weights for a category
    d_pipe = models_data["D_1_6_uncapped"]["pipeline"]
    d_tr = dict(d_pipe.named_steps["vectorizer"].transformer_list)
    d_w_vec = d_tr["word"]
    d_clf = d_pipe.named_steps["classifier"]
    ref_w_vocab = dict(models_data["ref_46k"]["pipeline"].named_steps["vectorizer"].transformer_list)["word"].vocabulary_

    def explain_new_ngrams(text: str, changed_mc_indices: list[int]) -> list[dict]:
        analyze_w = d_w_vec.build_analyzer()
        tokens = set(analyze_w(text))
        res = []
        for ci in changed_mc_indices:
            w = d_clf.estimators_[ci].coef_[0]
            items = []
            for tok in tokens:
                if tok in d_w_vec.vocabulary_ and tok not in ref_w_vocab:
                    idx = d_w_vec.vocabulary_[tok]
                    items.append((abs(float(w[idx])), float(w[idx]), tok, tok.count(" ") + 1))
            items.sort(reverse=True)
            res.append({
                "mcId": IDS[ci],
                "top_new_ngrams": [{"ngram": t, "len": ln, "weight": wt} for _, wt, t, ln in items[:6]],
            })
        return res

    # Find v2_test examples fixed by D (exact match in D, wrong in ref_46k), diverse across types
    fixed_v2_candidates = []
    for i, r in enumerate(v2_rows):
        ref_ok = np.array_equal(ref_v2_pred[i], v2_gold[i])
        d_ok = np.array_equal(d_v2_pred[i], v2_gold[i])
        if d_ok and not ref_ok:
            changed_cols = list(map(int, np.flatnonzero(ref_v2_pred[i] != d_v2_pred[i])))
            fixed_v2_candidates.append((r.get("type"), i, changed_cols))

    # Pick up to 12 diverse v2 examples + 2 public examples = 14 examples
    fixed_examples = []
    seen_types = collections.Counter()
    for typ, i, changed_cols in fixed_v2_candidates:
        if seen_types[typ] >= 2 and len(fixed_examples) >= 8:
            continue
        if len(fixed_examples) >= 12:
            break
        seen_types[typ] += 1
        r = v2_rows[i]
        fixed_examples.append({
            "dataset": "v2_test",
            "item_id": r["item_id"],
            "type": typ,
            "realism": v2_realism[i],
            "source_mcId": r.get("mcId", 101),
            "text": r["description"][:6000],
            "gold": [IDS[j] for j in np.flatnonzero(v2_gold[i])],
            "pred_ref_46k": [IDS[j] for j in np.flatnonzero(ref_v2_pred[i])],
            "pred_D_1_6": [IDS[j] for j in np.flatnonzero(d_v2_pred[i])],
            "prob_ref_46k": {str(IDS[j]): round(float(ref_v2_prob[i, j]), 4) for j in range(5)},
            "prob_D_1_6": {str(IDS[j]): round(float(d_v2_prob[i, j]), 4) for j in range(5)},
            "new_ngrams_explanation": explain_new_ngrams(r["description"][:6000], changed_cols),
        })

    for i, r in enumerate(pub_rows):
        if len(fixed_examples) >= 14:
            break
        if np.array_equal(d_pub_pred[i], pub_gold[i]) and not np.array_equal(ref_pub_pred[i], pub_gold[i]):
            changed_cols = list(map(int, np.flatnonzero(ref_pub_pred[i] != d_pub_pred[i])))
            fixed_examples.append({
                "dataset": "public_validation",
                "item_id": r.get("item_id", i),
                "type": "public",
                "realism": "public",
                "source_mcId": r.get("mcId", 101),
                "text": r["description"][:600],
                "gold": [IDS[j] for j in np.flatnonzero(pub_gold[i])],
                "pred_ref_46k": [IDS[j] for j in np.flatnonzero(ref_pub_pred[i])],
                "pred_D_1_6": [IDS[j] for j in np.flatnonzero(d_pub_pred[i])],
                "prob_ref_46k": {str(IDS[j]): round(float(ref_pub_prob[i, j]), 4) for j in range(5)},
                "prob_D_1_6": {str(IDS[j]): round(float(d_pub_prob[i, j]), 4) for j in range(5)},
                "new_ngrams_explanation": explain_new_ngrams(r["description"][:6000], changed_cols),
            })

    # Also check how many status_confirmation / status_negated_denial rows were resolved in vector space vs STILL predicted empty/wrong in D!
    status_resolved_vs_pred = {}
    for typ in ("status_confirmation", "status_negated_denial"):
        m = (v2_types == typ)
        idxs = np.flatnonzero(m)
        empty_ref = int((ref_v2_pred[m].sum(axis=1) == 0).sum())
        empty_d = int((d_v2_pred[m].sum(axis=1) == 0).sum())
        max_prob_d = d_v2_prob[m].max(axis=1)
        status_resolved_vs_pred[typ] = {
            "rows": int(m.sum()),
            "empty_ref_46k": empty_ref,
            "empty_D_1_6": empty_d,
            "max_prob_p50_D_1_6": float(np.median(max_prob_d)),
            "max_prob_p90_D_1_6": float(np.quantile(max_prob_d, 0.9)),
            "max_prob_max_D_1_6": float(np.max(max_prob_d)),
        }

    # Examples where D_1_6 did NOT help:
    # 1) A still-colliding vector pair in D_1_6 (from the 48 remaining vector collision groups)
    d_vec_fps = collisions_by_model["D_1_6_uncapped"]["fingerprints"]
    d_vec_groups = collections.defaultdict(list)
    for i, fp in enumerate(d_vec_fps):
        d_vec_groups[fp].append(i)
    d_any_vec_coll = [idxs for idxs in d_vec_groups.values() if len(idxs) > 1]

    failed_examples = []
    if d_any_vec_coll:
        idxs = d_any_vec_coll[0]
        a = idxs[0]
        b = next((i for i in idxs if not np.array_equal(v2_gold[i], v2_gold[a])), idxs[1])
        failed_examples.append({
            "kind": "still_colliding_vector_in_D_1_6",
            "type": v2_rows[a].get("type"),
            "augmentation": v2_rows[a].get("augmentation"),
            "source_mcId_a": v2_rows[a].get("mcId"),
            "source_mcId_b": v2_rows[b].get("mcId"),
            "item_a": v2_rows[a]["item_id"],
            "item_b": v2_rows[b]["item_id"],
            "offered_a": v2_rows[a]["labels"],
            "offered_b": v2_rows[b]["labels"],
            "gold_a": [IDS[j] for j in np.flatnonzero(v2_gold[a])],
            "gold_b": [IDS[j] for j in np.flatnonzero(v2_gold[b])],
            "pred_D_a": [IDS[j] for j in np.flatnonzero(d_v2_pred[a])],
            "pred_D_b": [IDS[j] for j in np.flatnonzero(d_v2_pred[b])],
            "prob_D_a": [round(float(x), 4) for x in d_v2_prob[a]],
            "prob_D_b": [round(float(x), 4) for x in d_v2_prob[b]],
            "text_a": v2_texts[a],
            "text_b": v2_texts[b],
        })

    # 2) Resolved-vector examples in D_1_6 from both status list types that still predict empty
    for typ in ("status_confirmation", "status_negated_denial"):
        for g_idx in baseline_group_tracking["D_1_6_uncapped"]["resolved_group_indices"]:
            idxs = baseline_231_groups[g_idx]
            a = idxs[0]
            if v2_rows[a].get("type") == typ and v2_gold[a].any() and not d_v2_pred[a].any():
                failed_examples.append({
                    "kind": "vector_separated_but_still_empty_in_D_1_6",
                    "type": typ,
                    "source_mcId": v2_rows[a].get("mcId"),
                    "item_id": v2_rows[a]["item_id"],
                    "gold": [IDS[j] for j in np.flatnonzero(v2_gold[a])],
                    "pred_ref_46k": [IDS[j] for j in np.flatnonzero(ref_v2_pred[a])],
                    "pred_D_1_6": [IDS[j] for j in np.flatnonzero(d_v2_pred[a])],
                    "prob_ref_46k": [round(float(x), 4) for x in ref_v2_prob[a]],
                    "prob_D_1_6": [round(float(x), 4) for x in d_v2_prob[a]],
                    "text": v2_texts[a],
                })
                break

    # 3) Regressions where ref_46k was exact and D_1_6 became wrong (one v2, one public)
    for typ_pref in ("rumour", "four_layers"):
        for i, r in enumerate(v2_rows):
            if r.get("type") == typ_pref and np.array_equal(ref_v2_pred[i], v2_gold[i]) and not np.array_equal(d_v2_pred[i], v2_gold[i]):
                failed_examples.append({
                    "kind": "regression_correct_in_ref_46k_wrong_in_D_1_6",
                    "type": r.get("type"),
                    "source_mcId": r.get("mcId"),
                    "item_id": r["item_id"],
                    "gold": [IDS[j] for j in np.flatnonzero(v2_gold[i])],
                    "pred_ref_46k": [IDS[j] for j in np.flatnonzero(ref_v2_pred[i])],
                    "pred_D_1_6": [IDS[j] for j in np.flatnonzero(d_v2_pred[i])],
                    "prob_ref_46k": [round(float(x), 4) for x in ref_v2_prob[i]],
                    "prob_D_1_6": [round(float(x), 4) for x in d_v2_prob[i]],
                    "text": v2_texts[i],
                })
                break

    for i, r in enumerate(pub_rows):
        if np.array_equal(ref_pub_pred[i], pub_gold[i]) and not np.array_equal(d_pub_pred[i], pub_gold[i]):
            failed_examples.append({
                "kind": "regression_public_correct_in_ref_46k_wrong_in_D_1_6",
                "type": "public",
                "source_mcId": r.get("mcId", 101),
                "item_id": r.get("item_id", i),
                "gold": [IDS[j] for j in np.flatnonzero(pub_gold[i])],
                "pred_ref_46k": [IDS[j] for j in np.flatnonzero(ref_pub_pred[i])],
                "pred_D_1_6": [IDS[j] for j in np.flatnonzero(d_pub_pred[i])],
                "prob_ref_46k": [round(float(x), 4) for x in ref_pub_prob[i]],
                "prob_D_1_6": [round(float(x), 4) for x in d_pub_prob[i]],
                "text": r["description"][:400],
            })
            break

    # --- 6. Correlation between collision reduction and F1 across the 5 configs ---
    coll_counts = [collisions_by_model[n]["summary"]["conflicting_src_vec_groups"] for n in CONFIG_NAMES]
    v2_f1s = [slice_evaluations[n]["v2_test"]["micro_f1"] for n in CONFIG_NAMES]
    status_f1s = [slice_evaluations[n]["v2_problem_lists"]["both_status_lists_1152"]["micro_f1"] for n in CONFIG_NAMES]
    coll642_f1s = [slice_evaluations[n]["v2_problem_lists"]["baseline_231_collision_642_rows"]["micro_f1"] for n in CONFIG_NAMES]
    corr_v2 = float(np.corrcoef(coll_counts, v2_f1s)[0, 1])

    report_data = {
        "configs": CONFIG_NAMES,
        "measurements": {n: models_data[n]["measurement"] for n in CONFIG_NAMES},
        "v2_collisions": {n: collisions_by_model[n]["summary"] for n in CONFIG_NAMES},
        "public_collisions": {n: pub_collisions_by_model[n]["summary"] for n in CONFIG_NAMES},
        "baseline_231_tracking": baseline_group_tracking,
        "ngrams_resolving_groups": ngrams_resolving_groups,
        "d_surviving_collisions": {
            "total_groups": len(d_remaining),
            "total_rows": sum(map(len, d_remaining)),
            "reason_counts": dict(surviving_reason_counts),
            "line_token_lengths_summary": {
                "min": int(min(surviving_line_token_lengths)) if surviving_line_token_lengths else 0,
                "median": float(np.median(surviving_line_token_lengths)) if surviving_line_token_lengths else 0.0,
                "max": int(max(surviving_line_token_lengths)) if surviving_line_token_lengths else 0,
            },
            "examples": d_surviving_diagnostics,
        },
        "slice_evaluations": slice_evaluations,
        "bootstrap_vs_ref": bootstrap_vs_ref,
        "flips_vs_ref": flips_vs_ref,
        "weight_analysis": weight_analysis,
        "status_resolved_vs_pred": status_resolved_vs_pred,
        "correlation_collisions_vs_v2_f1": {
            "conflicting_groups": coll_counts,
            "v2_micro_f1": v2_f1s,
            "status_lists_1152_f1": status_f1s,
            "collision_642_f1": coll642_f1s,
            "pearson_r_groups_vs_v2_f1": corr_v2,
        },
        "fixed_examples_count": len(fixed_v2_candidates),
        "fixed_examples": fixed_examples,
        "failed_examples": failed_examples,
    }

    out_path = exp_dir / "analysis.json"
    out_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Collisions (src_vec conflicting groups):", {n: collisions_by_model[n]["summary"]["conflicting_src_vec_groups"] for n in CONFIG_NAMES})
    print("Collisions (any vec groups):", {n: collisions_by_model[n]["summary"]["vector_collision_groups_any"] for n in CONFIG_NAMES})
    print("231 resolved:", {n: baseline_group_tracking[n]["resolved_no_conflict"] for n in CONFIG_NAMES})
    print("v2_test Micro-F1:", {n: round(slice_evaluations[n]["v2_test"]["micro_f1"], 6) for n in CONFIG_NAMES})
    print("status lists 1152 Micro-F1:", {n: round(slice_evaluations[n]["v2_problem_lists"]["both_status_lists_1152"]["micro_f1"], 6) for n in CONFIG_NAMES})
    print("status_resolved_vs_pred:", status_resolved_vs_pred)


if __name__ == "__main__":
    main()
