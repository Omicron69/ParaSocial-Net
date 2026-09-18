"""
Build the data-attrition funnel from the real audit + manifest logs.
Run from project root:  python attrition_from_audit.py
Reads:  data/interim/clean_audit.csv   (per-filter drop counts)
        data/interim/manifest.csv      (pre-cap vs capped totals)
Writes: data/processed/figures/attrition_funnel.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/root/tf-project/venv/Projects/BehaviouralAnalysis")
os.makedirs("data/processed/figures", exist_ok=True)

# ---- 1. filter drops from the audit ----
a = pd.read_csv("data/interim/clean_audit.csv")
kept  = int(a["kept"].sum())
bot   = int(a["no author/bot"].sum())
empty = int(a["removed/empty"].sum())
short = int(a["under 5 words"].sum())
dupe  = int(a["dupe text"].sum())
pre_filter = kept + bot + empty + short + dupe   # total entering the cleaning filters

# ---- 2. pre-cap raw total from the manifest ----
# manifest lists both raw rows (e.g. r_bangtan_comments) and capped rows
# (capped_r_bangtan_comments). The RAW collected total is the sum of the
# non-capped, non-NEW rows; the pre-filter figure above is the post-cap input.
m = pd.read_csv("data/interim/manifest.csv")
raw_rows = m[~m["file"].str.startswith("capped_") & ~m["file"].str.endswith("_NEW")]
raw_total = int(raw_rows["n"].sum())

capped_rows = m[m["file"].str.startswith("capped_")]
capped_total = int(capped_rows["n"].sum())

# ---- 3. assemble ordered stages (only those we can evidence) ----
# raw collected -> after 300k/sub cap -> minus bot -> minus empty
# -> minus <5 words -> minus dupes = cleaned kept
stages = [
    ("Raw collected",            raw_total),
    ("After 300k/sub cap",       capped_total),
    ("After bot / no-author",    capped_total - bot),
    ("After empty removed",      capped_total - bot - empty),
    ("After <5-word removed",    capped_total - bot - empty - short),
    ("After duplicate removed\n(cleaned dataset)", kept),
]
names = [s[0] for s in stages]
vals  = [s[1] for s in stages]

print("stage totals:")
for n, v in stages:
    print(f"  {n.splitlines()[0]:32} {v:>12,}")

# ---- 4. plot ----
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ypos = np.arange(len(names))[::-1]
maxv = vals[0]
for y, n, v in zip(ypos, names, vals):
    w = v / maxv
    ax.barh(y, w, left=(1-w)/2, height=0.6, color="steelblue", alpha=0.88)
    pct = v / maxv * 100
    ax.text(0.5, y, f"{n}\n{v:,}  ({pct:.0f}% of raw)", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(names)-0.4); ax.axis("off")
ax.set_title("Data attrition from raw collection to cleaned dataset", fontsize=13)
fig.tight_layout()
fig.savefig("data/processed/figures/attrition_funnel.png", dpi=150)
plt.close(fig)
print("\nsaved data/processed/figures/attrition_funnel.png")