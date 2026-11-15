"""Read-only lexical-neighbor diagnostic for the public split and two stress tests."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path('/srv/Hakaton')
REPO = ROOT / 'innopolis_2026'
OUT = ROOT / 'results/segmentation_v1/dataset_shift.json'


def read(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=['description']).column('description').to_pylist()


def norm(text):
    return ' '.join(text.casefold().split())


def main():
    model = joblib.load(ROOT/'models/baseline/tfidf_baseline.joblib')  # Trusted local file.
    vectorizer = model['pipeline'].named_steps['vectorizer']
    word = vectorizer.transformer_list[0][1]
    train = [row['description'] for row in read(ROOT/'data/processed/train.jsonl')]
    train_keys = {norm(text) for text in train}
    X = normalize(vectorizer.transform(train))
    queries = {
        'public_validation': [r['description'] for r in read(ROOT/'data/processed/validation.jsonl')],
        'public_test': parquet(REPO/'datasets/public/public-test-dataset.parquet'),
        'synthetic_300': parquet(REPO/'datasets/public/public-test-synthetic-300.parquet'),
        'new_long_test': [r['description'] for r in read(REPO/'datasets/postoyanka/splits/test.jsonl')],
    }
    summary = {}
    for name, texts in queries.items():
        nearest = []
        for start in range(0, len(texts), 64):
            V = normalize(vectorizer.transform(texts[start:start+64]))
            nearest.extend(np.asarray(cosine_similarity(V, X, dense_output=True).max(axis=1)).tolist())
        tokens = [word.build_tokenizer()(word.build_preprocessor()(text)) for text in texts]
        # Restrict to alphabetic token vocabulary; word TF-IDF also has bigrams.
        oov = [sum(token not in word.vocabulary_ for token in row)/len(row)
               for row in tokens if row]
        values = np.asarray(nearest)
        lengths = sorted(len(text) for text in texts)
        summary[name] = {
            'n': len(texts), 'median_chars': int(np.median(lengths)),
            'p90_chars': int(np.percentile(lengths,90)),
            'exact_normalized_overlap_with_public_train': sum(norm(t) in train_keys for t in texts),
            'nearest_train_cosine_median_p10_p90': [round(float(x),4) for x in np.percentile(values,[50,10,90])],
            'nearest_train_cosine_at_least_0_8': int((values>=.8).sum()),
            'median_word_token_oov_fraction': round(float(np.median(oov)),4),
        }
        print(name,summary[name],flush=True)
    OUT.write_text(json.dumps({'similarity_feature_space': 'existing word+char TF-IDF from trusted public-only model',
                               'disclaimer': 'Cosine similarity is lexical, not semantic; duplicates and near-duplicates can inflate public-test similarity',
                               'sets':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':
    main()
