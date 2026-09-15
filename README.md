# Behavioural Typology of Idol-Culture Engagement

Detecting parasocial, financial, and bullying harm in K-pop and J-pop fan communities through a four-phase machine-learning pipeline.

**MSc Artificial Intelligence dissertation — Kingston University London**
Author: Meraz (K2005941) · Supervisor: Dr. Pushpa Kumarapeli

## Overview

This project discovers behavioural typologies of fan engagement from ~3 million Reddit posts, detects three forms of harm using a domain-adapted language model, and links the two to quantify which behavioural patterns concentrate which harms.

**Headline finding:** harm modalities are *dissociable*. Intensely attached "Devoted" fans over-represent parasocial risk yet show low bullying and financial harm, while "Conflict-Exposed" and "Financially Intensive" users concentrate bullying and financial harm respectively.

The system is a **research prototype and first filter** for researchers and support services, not a clinical tool or an automated moderation system.

## Pipeline

| Phase | What it does |
|-------|--------------|
| A — Features | Multilingual behavioural lexicon + posting dynamics + DAPT embeddings → 53-feature per-user matrix |
| B — Clustering | PCA + k-means (and UMAP + HDBSCAN) → 7 behavioural typologies |
| C — Harm detection | Domain-adapted XLM-RoBERTa, multi-label, Focal loss |
| D — Linkage | Index Scores linking typologies to harm; SHAP explainability |

## Key results

- Domain-adaptive pre-training reduced perplexity **9.44 → 5.28 (44%)**.
- Harm classifier AUC: parasocial 0.73, bullying 0.88, financial 0.87.
- Outperformed a non-adapted baseline across all dimensions; generic toxicity tools could not represent parasocial or financial harm at all.
- Typology–harm pattern validated (label vs prediction r = 0.80–0.99) and robust across the full 2.26-million-post corpus.

## Repository structure

```
src/          lexicons.py, feats.py, clf.py
notebooks/    01_ingest ... 10_baselines_scale
harm_demo.py  interactive Gradio demo 
```

Data, trained models, and the anonymisation salt are not included, for privacy and file-size reasons.

## Running the demo

```
pip install -r requirements.txt
python harm_demo.py          # web UI (requires the trained model locally)
python harm_demo.py --cli    # terminal version
```

## Method notes

- Base model: XLM-RoBERTa, domain-adapted on the fan corpus (one epoch, 2.4M texts).
- Class imbalance: harm is rare (<10% of posts, <1% severe), addressed with Focal loss, per-label positive weighting, and harm-stratified annotation.
- Anonymisation: usernames were pseudonymised via salted SHA-256 at ingestion.

## Ethics

All source data were public Reddit posts, pseudonymised at ingestion. Raw data, annotated data, model weights, and the anonymisation salt are not published. The typology carries a dual-use tension: the same patterns that could help protect performers from coordinated harassment could be misused to target vulnerable fans commercially. Predatory applications are out of scope by design.

## Licence

All rights reserved (dissertation project).
- Typology–harm pattern validated (label vs prediction r = 0.80–0.99) and robust across the full 2.26-million-post corpus.

## Repository structure


