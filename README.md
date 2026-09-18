# Behavioural Typology of Idol-Culture Engagement

Detecting parasocial, financial, and bullying harm in K-pop and J-pop fan
communities through a four-phase machine-learning pipeline.

---

## Overview

This project discovers behavioural typologies of fan engagement from a large
corpus of K-pop and J-pop Reddit activity, detects three distinct forms of harm
using a domain-adapted language model, and links the two to quantify which
behavioural patterns concentrate which harms.

**Headline finding — the three harms are *distinguishable*, not a single
"at-risk" profile.** Intensely attached *Devoted* fans over-represent parasocial
risk yet show low bullying and financial harm, while *Conflict-Exposed* and
*Financially Intensive* users concentrate bullying and financial harm
respectively. Different behavioural types carry different specific harms, and the
pattern holds across the full 2.26-million-post corpus.

The system is a **research prototype and first filter** for researchers and
support services — not a clinical instrument and not an automated moderation
system. Predictions are decision support for a human, never automated action
against a user.

## Pipeline

| Phase | What it does |
|-------|--------------|
| **A — Features** | Multilingual behavioural lexicon + posting dynamics + DAPT embeddings → a 53-feature per-user matrix (33 behavioural + 20 embedding PCs) |
| **B — Clustering** | PCA + k-means (primary), corroborated by UMAP + HDBSCAN → 7 behavioural typologies |
| **C — Harm detection** | Domain-adapted XLM-RoBERTa, multi-label, three binary heads, Focal loss |
| **D — Linkage** | Index Scores linking typologies to harm; SHAP + lexicon highlighting for explainability |

## Data

Twenty subreddits collected via Arctic Shift, spanning K-pop and J-pop group
communities, member subs, snark/criticism subs, and a pop-culture control
(e.g. `bangtan`, `blackpink`, `twice`, `StrayKids`, `enhypen`, `AKB48`,
`nogizaka46`, `sakurazaka46`, `TWICEsnark`, `kpoprants`, `popculture`).

Attrition from collection to analysis:

```
~11.6M raw items  →  300k/subreddit cap → ~4.0M  →  cleaned 3,044,828
                  →  feature-eligible ~2.26M  →  39,396 users × 53 features
```

Usernames are pseudonymised via salted SHA-256 **at ingestion**; casing, emoji
and slang are preserved as behavioural signal.

## Behavioural typologies (k = 7)

| Typology | Users |
|----------|------:|
| Financially Intensive Collector | 8,660 |
| Casual Participant | 6,866 |
| Devoted | 5,969 |
| Conflict-Exposed | 5,937 |
| High-Intensity Multi-Domain | 4,670 |
| Peripheral | 4,460 |
| Broadly Engaged | 2,834 |

Confound check: the largest single-subreddit share of any cluster is 31%, so the
typologies reflect behaviour rather than which subreddit a user came from.

## Harm dimensions

Posts were hand-labelled on three independent dimensions:

- **parasocial_risk** — none / moderate / acute
- **bullying** — none / involved / targeted / victimised (records the poster's
  *role* in harm, not the mere presence of hostile words)
- **financial_harm** — none / moderate / acute

Severe tiers were too rare (<1%) to train reliably, so each dimension was
collapsed to a **binary** target (none vs. elevated) for the classifier, with the
severe sub-categories reported descriptively.

## Key results

- Domain-adaptive pre-training reduced perplexity **9.44 → 5.28 (44%)**.
- Harm classifier test AUC — parasocial **0.73**, bullying **0.88**,
  financial **0.87** (tuned for high recall as a first filter).
- Monotonic gains: off-the-shelf < non-adapted XLM-R < DAPT across all three
  dimensions; generic toxicity tools cannot represent parasocial or financial
  harm at all.
- Typology–harm linkage validated (label vs. prediction Index correlation
  r = 0.80–0.99) and robust on the full **2.26-million-post** corpus, e.g.
  Financially Intensive → financial, Conflict-Exposed → bullying,
  Devoted → parasocial.

## Repository structure

```
src/
  lexicons.py        multilingual harm lexicon (449 terms across 3 domains)
  feats.py           behavioural + DAPT-embedding feature extraction
  clf.py             harm classifier (3 binary heads on the DAPT encoder)
notebooks/
  01_ingest          Arctic Shift ingestion + salted-hash pseudonymisation
  02_clean           cleaning, per-subreddit capping, filtering
  03_oov             OOV analysis → curated fandom tokens
  04_dapt            domain-adaptive pre-training
  05_features        per-user feature matrix
  06_cluster         PCA+k-means, UMAP+HDBSCAN typologies
  07_annotate        manual annotation workflow
  08_classifier      classifier training + hyperparameter sweep
  09_index_shap      Index Scores + SHAP explainability
  10_baselines_scale baselines + full-corpus robustness
harm_demo.py         interactive Gradio demo
attrition_from_audit.py   data-attrition figures
dataset_table_gen.py      dataset table generation
requirements.txt
.gitignore
```

The dataset, annotated labels, trained model weights, and the anonymisation salt
are **not included** — for privacy and file-size reasons (see Ethics).

## Setup

Developed on WSL Ubuntu, Python 3.11, PyTorch 2.5.1 (CUDA 12.1). A GPU is
recommended for the model stages (developed on an RTX 4070 Super, 12 GB); DAPT
took roughly 65 hours for one epoch over 2.4M texts.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running the demo

```bash
python harm_demo.py
```

Launches a GOV.UK-styled Gradio web UI for single-post analysis — per-dimension
harm scores, a radar chart, typology context, and lexicon-term highlighting — as
well as batch scoring of a CSV. The demo requires the trained model to be present
locally; it is not distributed with the repository.

## Method notes

- **Base model:** XLM-RoBERTa, domain-adapted on the fan corpus (one epoch,
  2.4M texts); tokenizer extended with 313 curated fandom tokens (mean-init).
- **Classifier:** three binary heads on the DAPT encoder, Focal loss with
  per-label positive-class weighting; final settings lr = 2e-5, gamma = 1.0,
  4 epochs.
- **Class imbalance:** harm is rare (<10% of posts, <1% severe), addressed with
  Focal loss, positive weighting, and harm-stratified annotation.
- **Annotation:** ~2,500 posts labelled **entirely by hand** against a codebook
  (no automated pre-labelling); a single annotator, with the human label treated
  as the sole ground truth.

## Ethics

All source data were public Reddit posts, pseudonymised at ingestion via salted
SHA-256; reporting is at aggregate level throughout. Because the fan communities
are globally distributed, contributors fall under several data-protection regimes
at once (UK/EU GDPR, South Korea's PIPA, Japan's APPI, and others) with unknowable
per-user location; the project therefore relies on a legitimate-interests basis
with safeguards rather than infeasible cross-jurisdiction consent, applying UK
GDPR standards as a single strict floor.

The typology carries a genuine **dual-use tension**: the same patterns that could
help protect performers and fans from coordinated harassment could be misused to
target vulnerable fans commercially. Predatory and surveillance applications are
out of scope by design, and the released artefacts are aggregate-only and
code-only so they cannot be pointed at a named individual.

Raw data, annotated labels, model weights, and the anonymisation salt are not
published.

## Licence

All rights reserved (dissertation project).
