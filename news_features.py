"""
Weekly news-text features from Guardian articles (fetch_guardian.py), using a
small PRETRAINED sentence model for inference only (no training):
sentence-transformers/all-MiniLM-L6-v2, CPU.

Per article: embedding of "headline. summary" ->
  conflict score  = cosine similarity to conflict example sentences minus
                    similarity to cooperation example sentences (zero-shot)
  6 topic dims    = PCA of the embeddings, fit on TRAINING-period articles only
Per week (Monday-Sunday, same calendar as the GDELT panel):
  dyad features    articles tagged with both countries (pair co-coverage)
  country features articles tagged with the country
  each = [log1p(article count), mean conflict score, mean of the 6 PCA dims]

    python news_features.py
Output:
    raw data/news_features.npz   dyad (D, T, 8), country (C, T, 8), aligned to load_panel()
"""

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from honest_common import GOLDSTEIN_VAL_START, MENA, load_panel

ARTICLES = "raw data/guardian_articles.parquet"
OUT = "raw data/news_features.npz"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_PCA = 6

CONFLICT = ["Airstrikes killed civilians", "The army launched a military offensive", "Rockets were fired across the border",
            "The government threatened retaliation", "Fighting broke out between forces", "Sanctions were imposed after the attack",
            "Troops were deployed to the border", "Protesters clashed with security forces"]
COOPERATION = ["The two countries signed an agreement", "Leaders met for peace talks", "A ceasefire was agreed",
               "The ministers announced economic cooperation", "Diplomatic relations were restored",
               "Aid was delivered to the region", "The sides agreed to resume negotiations", "A trade deal was announced"]


def clean(s):
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def main():
    from sentence_transformers import SentenceTransformer
    art = pd.read_parquet(ARTICLES)
    art = art[art.countries != ""].reset_index(drop=True)
    texts = (art.headline.map(clean) + ". " + art.summary.map(clean)).tolist()
    print(f"{len(art)} articles; embedding with {MODEL} on CPU...", flush=True)
    model = SentenceTransformer(MODEL, device="cpu")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    anchors = model.encode(CONFLICT + COOPERATION, normalize_embeddings=True)
    conflict = emb @ anchors[:len(CONFLICT)].T
    coop = emb @ anchors[len(CONFLICT):].T
    score = conflict.mean(1) - coop.mean(1)

    p = load_panel()
    weeks = p["weeks"]
    date = pd.to_datetime(art.date)
    monday = (date - pd.to_timedelta(date.dt.weekday, unit="D")).dt.normalize()
    t_idx = weeks.get_indexer(monday)
    train_mask = (monday < pd.Timestamp(GOLDSTEIN_VAL_START)).to_numpy()
    pca = PCA(N_PCA).fit(emb[train_mask])
    comp = pca.transform(emb)
    print(f"PCA on {train_mask.sum()} training-period articles explains {pca.explained_variance_ratio_.sum():.0%}")
    per_article = np.column_stack([score, comp])  # (N, 7)

    C, D, T = len(MENA), len(p["a1"]), len(weeks)
    ci = {c: i for i, c in enumerate(MENA)}
    di = {(a, b): i for i, (a, b) in enumerate(zip(p["a1"], p["a2"]))}
    sums_c, n_c = np.zeros((C, T, 7)), np.zeros((C, T))
    sums_d, n_d = np.zeros((D, T, 7)), np.zeros((D, T))
    for k, (cs, t) in enumerate(zip(art.countries.str.split("|"), t_idx)):
        if t < 0:
            continue
        for c in cs:
            sums_c[ci[c], t] += per_article[k]
            n_c[ci[c], t] += 1
        for a in cs:  # co-coverage feeds both directions of every pair; single-country -> domestic dyad
            for b in cs:
                if a != b or len(cs) == 1:
                    sums_d[di[(a, b)], t] += per_article[k]
                    n_d[di[(a, b)], t] += 1

    def finish(sums, n):
        mean = np.where(n[..., None] > 0, sums / np.maximum(n, 1)[..., None], 0.0)
        return np.concatenate([np.log1p(n)[..., None], mean], axis=2).astype("float32")

    np.savez(OUT, dyad=finish(sums_d, n_d), country=finish(sums_c, n_c), countries=np.array(MENA),
             weeks=np.array(weeks.date, dtype="datetime64[D]"))
    print(f"saved {OUT}: dyad-weeks with >=1 article {np.mean(n_d > 0):.1%}, country-weeks {np.mean(n_c > 0):.1%}")
    # sanity: ISR-PSE conflict score around 7 Oct 2023
    d = di[("ISR", "PSE")]
    for s in ["2023-09-18", "2023-09-25", "2023-10-02", "2023-10-09", "2023-10-16", "2023-11-27"]:
        t = weeks.get_loc(pd.Timestamp(s))
        print(f"  ISR-PSE {s}: articles {int(n_d[d, t]):>3}  conflict score {finish(sums_d, n_d)[d, t, 1]:+.3f}")


if __name__ == "__main__":
    main()
