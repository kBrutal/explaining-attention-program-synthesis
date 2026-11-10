"""M15: paper figures from the cached results → figures/*.png. Skips figures whose inputs don't exist yet.

Run: python -m pipeline.plots
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from pipeline.baselines import baselines  # noqa: E402
from pipeline.categorize import CATEGORIES  # noqa: E402
from pipeline.crossmatch import best_programs, crossmatch, library  # noqa: E402
from pipeline.extract import load_attention, run_dir  # noqa: E402
from pipeline.sandbox import run_program  # noqa: E402

OUT = Path("figures")
# validated categorical slots (dataviz reference palette, light mode), assigned in fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY, INK, MUTED = "#8a8a86", "#1f1f1e", "#6b6b67"
MODELS = {"gpt2": "GPT-2", "bert-base-uncased": "BERT-base"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "figure.dpi": 150})


def own_iou(model, split="stories"):
    s = crossmatch(model)
    return np.array([s[split][p, l, h] for p, (l, h) in enumerate(s["heads"])])


def fig3():
    """Box plots: intended / best programs vs baselines, on whole stories."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (m, name) in zip(axes, MODELS.items()):
        b = baselines(m)
        groups = {"Intended": own_iou(m), "Best": best_programs(m)[1].ravel()}
        groups |= {k.split("/")[1].replace(" (extra)", "*"): v.ravel() for k, v in b.items() if k.startswith("stories/")}
        bp = ax.boxplot(list(groups.values()), patch_artist=True, widths=0.55, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5})
        for i, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=SLOTS[0] if i < 2 else "#d9d9d5", edgecolor="white", linewidth=1)
        ax.set_xticks(range(1, len(groups) + 1), list(groups), rotation=30, ha="right")
        ax.set_title(f"{name} (whole stories)", color=INK, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("IoU with real attention (per head)")
    fig.text(0.01, 0.01, "* extra baseline, not in the paper", color=MUTED, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_iou_vs_baselines.png")


def fig4():
    """(a) best-fit IoU per head, each layer sorted; (b) category of each head's best program, each layer sorted."""
    for m, name in MODELS.items():
        _, iou = best_programs(m)
        cats = json.loads((run_dir(m) / "categories.json").read_text())["heads"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
        im = a.imshow(-np.sort(-iou, 1).T, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
        fig.colorbar(im, ax=a, label="best-fit IoU")
        a.set(xlabel="layer", ylabel="heads (sorted within layer)", title=f"(a) {name} best-fit IoU")
        idx = np.array([sorted(CATEGORIES.index(c) for c in row) for row in cats])
        cmap = ListedColormap(SLOTS)
        b.imshow(idx.T, cmap=cmap, vmin=-0.5, vmax=5.5, aspect="auto", origin="lower", interpolation="nearest")
        b.set(xlabel="layer", title=f"(b) {name} head categories")
        b.legend(handles=[Patch(color=SLOTS[i], label=c) for i, c in enumerate(CATEGORIES)],
                 bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        for ax in (a, b):
            ax.grid(False)
            ax.set_xticks(range(12))
        fig.tight_layout()
        fig.savefig(OUT / f"fig4_{m}_heads.png")


def fig2(model="gpt2", k=0):
    """Real vs predicted attention for the best- and worst-fit intended programs on one story."""
    s, recs = crossmatch(model), library(model)
    own = own_iou(model)
    attn = load_attention(model, "stories")[k]
    picks = [int(np.argmax(own)), int(np.argsort(own)[len(own) // 2])]
    n = min(40, len(attn[0]))
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for row, p in zip(axes, picks):
        l, h = s["heads"][p]
        P = run_program(model, recs[p]["candidates"][recs[p]["chosen"]]["src"], "stories")[0][k]
        for ax, M, label in ((row[0], attn[1][l, h], "real"), (row[1], P, "program")):
            ax.imshow(M[:n, :n] if M is not None else np.zeros((n, n)), cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"L{l}H{h} {label}" + (f" (IoU {own[p]:.2f} over stories)" if label == "program" else ""),
                         loc="left", color=INK)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{MODELS[model]}: best and median intended program, first {n} tokens of a story", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_examples.png")


def fig5():
    path = run_dir("gpt2") / "ppl.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    x = 100 * np.array(r["fraction"])
    inc = {c: np.array(v) for c, v in r["increase_pct"].items()}
    wrong = np.stack([v for c, v in inc.items() if c.startswith("wrong")])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    series = [("chosen", inc["chosen"], SLOTS[0]), ("wrong category", wrong.mean(0), SLOTS[1]),
              ("pruning", inc["prune"], SLOTS[2])]
    for label, y, c in series:  # level 0 is 0% by definition: not drawable on a log axis
        a.plot(x[1:], np.maximum(y[1:], 1e-2), color=c, linewidth=2, label=label)
    a.fill_between(x[1:], np.maximum(wrong.min(0)[1:], 1e-2), np.maximum(wrong.max(0)[1:], 1e-2),
                   color=SLOTS[1], alpha=0.15, linewidth=0)
    a.set(yscale="log", xlabel="% of heads replaced", ylabel="perplexity increase %", title="(a) GPT-2 perplexity")
    a.legend(frameon=False, loc="upper left")
    xs = np.array(r["mean_iou_replaced"][1:], float)
    ys = inc["chosen"][1:]
    ok = ys > 0
    b.scatter(xs[ok], ys[ok], s=18, color=SLOTS[0], edgecolor="white", linewidth=0.8)
    b.set(yscale="log", xlabel="mean IoU of replaced heads", ylabel="perplexity increase %",
          title=f"(b) chosen condition, Pearson r = {r['fig5b_pearson_r']:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_perplexity.png")


def fig6():
    path = run_dir("gpt2") / "bench.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    x = 100 * np.array(r["levels"])
    acc = r["acc"]
    benches = list(acc["chosen"])
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharex=True)
    for ax, bench in zip(axes.ravel(), benches):
        wrong = np.stack([v[bench] for c, v in acc.items() if c.startswith("wrong")])
        for label, y, c in (("chosen", acc["chosen"][bench], SLOTS[0]), ("wrong category", wrong.mean(0), SLOTS[1]),
                            ("pruning", acc["prune"][bench], SLOTS[2])):
            ax.plot(x, 100 * np.array(y), color=c, linewidth=2, marker="o", markersize=3, label=label)
        ax.fill_between(x, 100 * wrong.min(0), 100 * wrong.max(0), color=SLOTS[1], alpha=0.15, linewidth=0)
        ax.set_title(bench, loc="left", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("% of heads replaced")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy %")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("GPT-2 zero-shot accuracy (200 examples each)", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_benchmarks.png")


if __name__ == "__main__":  # required: sandbox children re-import the main module
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig2, fig3, fig4, fig5, fig6):
        f()
    print("figures →", sorted(p.name for p in OUT.glob("*.png")))
