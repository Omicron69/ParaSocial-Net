"""
Idol-Fandom Harm Classifier (research prototype).

Bridges the Phase C harm classifier with Phase D typology findings. Analyse a
single post, or upload a CSV for batch scoring.

USAGE
    python harm_demo.py            web UI (recommended)
    python harm_demo.py --cli      terminal version, no extra install

REQUIREMENTS (web UI): pip install gradio pandas   (matplotlib already installed)
"""
import sys
import tempfile
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "models/dapt_final"
WEIGHTS   = "models/harm_clf_final.pt"
DIMS      = ["Parasocial risk", "Bullying", "Financial harm"]
THRESHOLD = 0.5
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

HARM_TYPOLOGY = {
    "Parasocial risk": "This linguistic pattern is most concentrated in Devoted users (Index 128.3), who show deep emotional attachment and dependency.",
    "Bullying":        "This linguistic pattern is most concentrated in Conflict-Exposed users (Index 240.0), associated with cross-fandom friction and snark spaces.",
    "Financial harm":  "This linguistic pattern is most concentrated in Financially Intensive Collector users (Index 291.9), associated with heavy merchandise and photocard spending.",
}

# GOV.UK-style palette
GOVUK_BLUE   = "#1d70b8"
GOVUK_GREEN  = "#00703c"
GOVUK_RED    = "#d4351c"
GOVUK_YELLOW = "#ffdd00"
GOVUK_GREY   = "#505a5f"
GOVUK_BLACK  = "#0b0c0c"


class HarmClassifier(nn.Module):
    def __init__(self, encoder, n_labels=3, extra_dim=0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size + extra_dim, n_labels)

    def forward(self, ids, mask, extra=None):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=1)
        return self.head(pooled)


print("Loading model. First run takes a few seconds.")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = HarmClassifier(AutoModel.from_pretrained(MODEL_DIR))
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE).eval()
print(f"Model ready. Running on {DEVICE}.")


def classify(text):
    """UNCHANGED model logic. Returns {dimension: probability}."""
    if not text or not str(text).strip():
        return {d: 0.0 for d in DIMS}
    enc = tok(str(text), truncation=True, max_length=192, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(enc.input_ids, enc.attention_mask))[0].cpu().tolist()
    return {d: round(p, 3) for d, p in zip(DIMS, probs)}


def bars_html(probs):
    """GOV.UK-style horizontal percentage bars."""
    rows = []
    for d, p in probs.items():
        pct = int(round(p * 100))
        colour = GOVUK_RED if p >= THRESHOLD else GOVUK_BLUE
        rows.append(f"""
        <div style="margin-bottom:14px;font-family:Arial,sans-serif;">
          <div style="display:flex;justify-content:space-between;font-size:16px;
                      color:{GOVUK_BLACK};margin-bottom:4px;">
            <span>{d}</span><span style="font-weight:bold;">{pct}%</span>
          </div>
          <div style="background:#f3f2f1;height:22px;border:1px solid #b1b4b6;">
            <div style="width:{pct}%;height:100%;background:{colour};"></div>
          </div>
        </div>""")
    return f'<div>{"".join(rows)}</div>'


# lexicon term highlighting (explainability). Uses the project's own lexicons.
import re as _re, html as _html
try:
    import sys as _sys
    _sys.path.append("src")
    from lexicons import DOMAINS as _DOMAINS, terms as _terms
    _LEXMAP = {"Parasocial risk": _terms(_DOMAINS["parasocial"]),
               "Bullying": _terms(_DOMAINS["victim"]),
               "Financial harm": _terms(_DOMAINS["financial"])}
except Exception:
    _LEXMAP = {"Parasocial risk": [], "Bullying": [], "Financial harm": []}

_HL_COLOUR = {"Parasocial risk": "#fff2cc", "Bullying": "#f8d7da", "Financial harm": "#d1e7dd"}


def highlight(text, flagged_dims):
    """Highlight lexicon terms for the flagged harm dimensions."""
    out = _html.escape(text)
    for dim in flagged_dims:
        for term in sorted(_LEXMAP.get(dim, []), key=len, reverse=True):
            if len(term) < 3:
                continue
            pat = _re.compile(_re.escape(term), _re.IGNORECASE)
            out = pat.sub(lambda m: f'<mark style="background:{_HL_COLOUR[dim]};padding:0 2px;">{m.group(0)}</mark>', out)
    legend = ('<div style="font-size:13px;color:#505a5f;margin-top:8px;">'
              'Highlighted: '
              '<mark style="background:#fff2cc;">parasocial</mark> '
              '<mark style="background:#f8d7da;">bullying</mark> '
              '<mark style="background:#d1e7dd;">financial</mark> '
              'lexicon terms. Highlighting shows lexicon matches, not the full model reasoning.</div>')
    return f'<div style="font-size:16px;line-height:1.7;">{out}</div>{legend}'


def radar(probs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = list(probs.keys())
    vals = list(probs.values())
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    ax.plot(ang, vals, color=GOVUK_BLUE, linewidth=2)
    ax.fill(ang, vals, color=GOVUK_BLUE, alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def process_batch(file):
    import pandas as pd
    if file is None:
        return None
    df = pd.read_csv(file.name)
    text_col = None
    for cand in ["text", "post", "Text", "Post", "content", "body"]:
        if cand in df.columns:
            text_col = cand; break
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c; break
    if text_col is None:
        raise ValueError("No text column found in the CSV.")
    para, bully, fin = [], [], []
    for t in df[text_col]:
        p = classify(t)
        para.append(p["Parasocial risk"]); bully.append(p["Bullying"]); fin.append(p["Financial harm"])
    df["Parasocial_score"] = para
    df["Bullying_score"] = bully
    df["Financial_score"] = fin
    out = tempfile.NamedTemporaryFile(delete=False, suffix="_scored.csv")
    df.to_csv(out.name, index=False)
    return out.name


# GOV.UK-style CSS
GOVUK_CSS = """
.gradio-container {font-family: Arial, Helvetica, sans-serif !important;}
h1, h2, h3 {color: #0b0c0c !important; font-weight: 700 !important;}
.govuk-header {background:#0b0c0c; padding:14px 20px; margin-bottom:20px;}
.govuk-header span {color:#ffffff; font-size:20px; font-weight:700;}
button.primary {background:#00703c !important; border:none !important;
    box-shadow:0 2px 0 #002d18 !important; color:#fff !important;}
.govuk-inset {border-left:5px solid #b1b4b6; padding:12px 18px; margin:10px 0;
    background:#f8f8f8;}
"""


def web():
    import gradio as gr

    def predict(text):
        if not text or not text.strip():
            return "", "", None, "Enter some text above, then select Analyse."
        probs = classify(text)
        flags = [d for d, p in probs.items() if p >= THRESHOLD]
        hl = highlight(text, flags) if flags else highlight(text, list(probs.keys()))
        if flags:
            lines = ["Result: flagged for " + ", ".join(flags) + ".", ""]
            for d in flags:
                lines.append(HARM_TYPOLOGY[d])
            lines.append("")
            lines.append("This is a first-filter research prototype. Flagged posts "
                         "would be passed to a person for review, not actioned "
                         "automatically.")
            verdict = "\n\n".join(lines)
        else:
            verdict = "Result: no harm flagged above the 50% threshold."
        return hl, bars_html(probs), radar(probs), verdict

    with gr.Blocks(theme=gr.themes.Base(), css=GOVUK_CSS,
                   title="Idol-Fandom Harm Classifier") as demo:
        gr.HTML('<div class="govuk-header"><span>Idol-Fandom Harm Classifier '
                '(research prototype)</span></div>')
        gr.Markdown(
            "This tool scores a fan post for three kinds of potential harm and links "
            "each to the behavioural typology it is most associated with. It is a "
            "research prototype intended as a first filter for human review. It is not "
            "a clinical tool or an automated moderation system."
        )

        with gr.Tab("Single post analysis"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp = gr.Textbox(lines=4, label="Fan post or tweet",
                                     placeholder="Paste the text to analyse.")
                    btn = gr.Button("Analyse", variant="primary")
                with gr.Column(scale=2):
                    out_bars = gr.HTML(label="Harm probabilities")
                    out_plot = gr.Plot(label="Harm profile")
            out_highlight = gr.HTML(label="Text with lexicon terms highlighted")
            out_verdict = gr.Markdown()
            gr.Examples(
                examples=[
                    ["my bias is literally my only reason to keep going honestly"],
                    ["these delusional stans need to be stopped, dragging them all"],
                    ["spent my entire paycheck on photocards again, can't afford rent"],
                    ["loved the comeback stage, they did amazing tonight"],
                    ["found her address, everyone go report this account"],
                ],
                inputs=inp,
                label="Examples with typical typology. "
                      "1 Devoted. 2 Conflict-Exposed. 3 Financially Intensive Collector. "
                      "4 Casual Participant. 5 Conflict-Exposed, targeted.",
            )
            btn.click(predict, inputs=inp, outputs=[out_highlight, out_bars, out_plot, out_verdict])
            inp.submit(predict, inputs=inp, outputs=[out_highlight, out_bars, out_plot, out_verdict])

        with gr.Tab("Batch process CSV"):
            gr.Markdown(
                "Upload a CSV containing a text column. The column named text or post "
                "is used if present, otherwise the first text column is used. Three "
                "score columns are added and the scored file is returned for download."
            )
            csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_btn = gr.Button("Process batch", variant="primary")
            csv_out = gr.File(label="Download scored CSV")
            batch_btn.click(process_batch, inputs=csv_in, outputs=csv_out)

        with gr.Accordion(label="Typology glossary", open=False):
            gr.Markdown(
                "**Broadly Engaged.** Users interacting across multiple domains without extreme concentration in one.\n\n"
                "**Casual Participant.** The baseline user. Low risk, primarily general pop-culture discussion.\n\n"
                "**Conflict-Exposed.** Highly over-represents bullying (Index 240.0). Frequent cross-fandom friction and snark spaces.\n\n"
                "**Devoted.** Over-represents parasocial harm (Index 128.3). Deep emotional attachment and dependency.\n\n"
                "**Financially Intensive Collector.** Over-represents financial harm (Index 291.9). Heavy merchandise and photocard spending.\n\n"
                "**High-Intensity Multi-Domain.** High-volume posters with elevated engagement across all metrics.\n\n"
                "**Peripheral.** Low-frequency posters on the edges of the community network."
            )

    demo.launch(share=True)


def cli():
    print("Paste a fan post or tweet. Blank line to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        probs = classify(text)
        for d, p in probs.items():
            print(f"  {d:16} {p*100:5.1f}%  {'#'*int(p*20)}")
        flags = [d for d, p in probs.items() if p >= THRESHOLD]
        print("  Flagged:", ", ".join(flags) if flags else "none", "\n")


if __name__ == "__main__":
    cli() if "--cli" in sys.argv else web()