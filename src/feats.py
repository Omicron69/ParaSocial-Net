"""
Phase A feature extraction.

Two feature blocks per user:
  1. Behavioural  — lexicon densities (parasocial/financial/victim) + posting
                    dynamics (rate, burstiness, late-night, sentiment volatility).
  2. Embedding    — mean-pooled DAPT sentence embeddings (added separately in the
                    notebook, on GPU).

Rate features are normalised by each user's own active span, so the wide
2013-2026 range across subreddits doesn't inflate users from short-lived subs.
"""

import numpy as np
import pandas as pd
from lexicons import DOMAINS, terms

LEX = {dom: {g: t for g, t in groups.items()} for dom, groups in DOMAINS.items()}


def count_hits(text, term_list):
    return sum(text.count(t) for t in term_list)


def score_post(text):
    words = max(len(text.split()), 1)
    out = {}
    for dom, groups in LEX.items():
        for g, tlist in groups.items():
            out[f"{dom}_{g}"] = count_hits(text, tlist) / words * 100
    out["caps"] = sum(c.isupper() for c in text) / max(len(text), 1) * 100
    out["excl"] = text.count("!") / words * 100
    out["first_person"] = count_hits(text, [" i ", " my ", " me ", " mine "]) / words * 100
    out["nwords"] = words
    return out


def user_features(g):
    g = g.sort_values("dt")
    gaps = g.dt.diff().dt.total_seconds().dropna() / 3600
    span_days = max((g.dt.max() - g.dt.min()).days, 1)
    hour = g.dt.dt.hour

    f = {}
    lex_cols = [c for c in g.columns if any(c.startswith(d + "_") for d in LEX)]
    for c in lex_cols:
        f[f"{c}_mean"] = g[c].mean()
    f["parasocial_peak"] = g[[c for c in lex_cols if c.startswith("parasocial")]].sum(1).max()
    f["financial_peak"] = g[[c for c in lex_cols if c.startswith("financial")]].sum(1).max()
    f["victim_peak"] = g[[c for c in lex_cols if c.startswith("victim")]].sum(1).max()

    f["caps_mean"] = g.caps.mean()
    f["excl_mean"] = g.excl.mean()
    f["firstperson_mean"] = g.first_person.mean()
    f["len_mean"] = g.nwords.mean()

    f["n_posts"] = len(g)
    f["rate"] = len(g) / span_days
    f["gap_med"] = gaps.median() if len(gaps) else np.nan
    f["gap_cv"] = (gaps.std() / gaps.mean()) if len(gaps) > 1 and gaps.mean() else np.nan
    f["burst"] = (gaps < 0.25).mean() if len(gaps) else 0.0
    f["latenight"] = hour.between(1, 5).mean()
    f["active_days_frac"] = g.dt.dt.floor("D").nunique() / span_days
    f["span_days"] = span_days
    f["n_subs"] = g.subreddit.nunique()

    if "sent" in g:
        f["sent_mean"] = g.sent.mean()
        f["sent_vol"] = g.sent.std()

    mid = g.dt.iloc[len(g) // 2] if len(g) > 3 else None
    if mid is not None:
        a, b = g[g.dt < mid], g[g.dt >= mid]
        for dom in ("parasocial", "financial", "victim"):
            cols = [c for c in lex_cols if c.startswith(dom)]
            if len(a) and len(b):
                f[f"{dom}_escalation"] = b[cols].sum(1).mean() - a[cols].sum(1).mean()
    return pd.Series(f)


def ihs(x):
    return np.log(x + np.sqrt(x ** 2 + 1))


def transform(U):
    U = U.fillna(U.median(numeric_only=True))
    Ui = U.apply(ihs)
    rng = (Ui.max() - Ui.min()).replace(0, 1)
    return (Ui - Ui.min()) / rng * 100
