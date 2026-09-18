# Dataset table from the manifest. Run from project root.
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir("/root/tf-project/venv/Projects/BehaviouralAnalysis")

m = pd.read_csv("data/interim/manifest.csv")
cap = m[m.file.str.startswith("capped_")].copy()   # capped = what was kept

sub_meta = {   # subreddit -> (genre, group)
 "AKB48":("J-pop","girl"),"Nogizaka46":("J-pop","girl"),"Sakurazaka46":("J-pop","girl"),
 "jpop":("J-pop","mixed"),"bangtan":("K-pop","boy"),"StrayKids":("K-pop","boy"),
 "enhypen":("K-pop","boy"),"BlackPink":("K-pop","girl"),"twice":("K-pop","girl"),
 "nayeon":("K-pop","girl"),"jihyo":("K-pop","girl"),"TwiceSnark":("K-pop","girl"),
 "kpoprants":("K-pop","mixed"),"kpop_uncensored":("K-pop","mixed"),
 "kpoptrulyuncensored":("K-pop","mixed"),"kpopharshopinions":("K-pop","mixed"),
 "unpopularkpopopinions":("K-pop","mixed"),"kpopcollections":("K-pop","mixed"),
 "popculture":("control","n/a"),
}

g = cap.groupby("subreddit").n.sum().sort_values(ascending=False)
rows, total = [], 0
for sub, n in g.items():
    genre, grp = sub_meta.get(sub, ("K-pop","mixed"))
    rows.append([sub, genre, grp, f"{int(n):,}"]); total += int(n)
rows.append(["Total","","", f"{total:,}"])

fig, ax = plt.subplots(figsize=(8, 0.42*len(rows)+1)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=["Subreddit","Genre","Group","Items"],
             cellLoc="left", loc="center", colWidths=[0.4,0.2,0.2,0.2])
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.5)
for j in range(4):
    t[0,j].set_facecolor("#33415C"); t[0,j].set_text_props(color="white", fontweight="bold")
for j in range(4):
    t[len(rows),j].set_text_props(fontweight="bold")
ax.set_title("Table 5.1: Collected dataset by subreddit", fontweight="bold", pad=12)
fig.tight_layout(); fig.savefig("data/processed/figures/dataset_table.png", dpi=150, bbox_inches="tight")
print("total:", f"{total:,}")