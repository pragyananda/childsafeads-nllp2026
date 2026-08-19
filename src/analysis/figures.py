"""Publication figures for the system design report.

Two figures, in the order a reader should meet them:

  fig1  what each data level buys, per sub-task -- the paper's thesis. The story
        is the divergence: ST2 climbs with the crawl, ST3 falls.
  fig2  the cascade cost curve -- how little of the corpus actually needs
        crawling, and that choosing well beats crawling more.
  fig3  the routed system -- which model family answers which sub-task, and at
        what data cost.
  fig4  the validation result -- four of five adoption calls correct, and the
        anatomy of the fifth.
  fig3  the routed system: which model family answers which sub-task, and at
        what data cost.
  fig4  the validation result -- four of five adoption calls correct, and the
        anatomy of the fifth.

Palette: categorical slots 1-3 of the validated reference palette, assigned in
fixed order (ST1, ST2, ST3). Verified with the skill's validator on the light
surface, all-pairs mode: worst CVD dE 9.2, worst normal-vision dE 24.0. Aqua
sits below 3:1 contrast, which the direct labels relieve.

Single light theme is deliberate: these are print figures for an ACL-style PDF,
not a themed web page.

Output: vector PDF (for LaTeX) plus PNG at 200dpi (for previews).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "documents" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SERIES = {"ST1": "#2a78d6", "ST2": "#eb6834", "ST3": "#1baf7a"}
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def style(ax):
    """Recessive axes: no top/right spines, hairline horizontal grid only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- figure 1
# Dev macro-F1 by access level, mean +/- sd across independent runs.
# L2 was run once, so it carries no error bar and is drawn hollow.
LEVELS = ["L1\ntranscript", "L2\n+ metadata", "L4\n+ product page"]
DATA = {
    "ST1": ([0.604, 0.594, 0.686], [0.013, None, 0.059]),
    "ST2": ([0.621, 0.637, 0.749], [0.014, None, 0.014]),
    "ST3": ([0.427, 0.416, 0.396], [0.021, None, 0.027]),
}
LABEL = {"ST1": "ST1  commercial type", "ST2": "ST2  product category",
         "ST3": "ST3  compliance flags"}

fig, ax = plt.subplots(figsize=(6.4, 4.0))
style(ax)
x = range(len(LEVELS))
for name, (ys, es) in DATA.items():
    c = SERIES[name]
    ax.plot(x, ys, color=c, linewidth=2, zorder=3, solid_capstyle="round")
    for xi, (y, e) in enumerate(zip(ys, es)):
        if e is None:      # single run: hollow marker, no error bar
            ax.plot(xi, y, "o", mfc="white", mec=c, mew=2, ms=8, zorder=4)
        else:
            ax.errorbar(xi, y, yerr=e, color=c, capsize=3, capthick=1.2,
                        elinewidth=1.2, zorder=3)
            ax.plot(xi, y, "o", color=c, ms=8, zorder=4)
    # Direct label in ink, adjacent to the coloured line end.
    ax.annotate(LABEL[name], (len(LEVELS) - 1, ys[-1]), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=INK, fontweight="medium")

ax.annotate("+0.128\np < 0.0001", (2, 0.749), xytext=(-118, 6),
            textcoords="offset points", fontsize=8.5, color=INK_SOFT, ha="center",
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9,
                            connectionstyle="arc3,rad=-0.2"))
ax.annotate("−0.031\np = 0.018", (2, 0.396), xytext=(-118, -14),
            textcoords="offset points", fontsize=8.5, color=INK_SOFT, ha="center",
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9,
                            connectionstyle="arc3,rad=0.2"))

ax.set_xticks(list(x)); ax.set_xticklabels(LEVELS, fontsize=9)
ax.set_xlim(-0.25, 2.95)
ax.set_ylim(0.33, 0.80)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_ylabel("dev macro-F1")
ax.set_xlabel("data access level  (increasing collection cost →)", labelpad=8)
ax.set_title("The three sub-tasks disagree about what data they want",
             loc="left", fontsize=11, fontweight="bold", pad=12)
ax.annotate("Mean ± sd over independent runs (n = 9, 1, 8). L2 ran once, so it is drawn hollow, without an interval.",
            (0, 0), xytext=(0, -58), textcoords="offset points",
            xycoords="axes fraction", fontsize=7.5, color=INK_MUTED)
fig.subplots_adjust(left=0.10, right=0.72, top=0.88, bottom=0.26)
fig.savefig(OUT / "fig1_data_levels.pdf")
fig.savefig(OUT / "fig1_data_levels.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
BUDGET = [0, 10, 20, 30, 50, 70, 100]
POLICY = {
    "targeted by model uncertainty": ("#2a78d6", [.540, .577, .588, .582, .590, .594, .596]),
    "compliance-flagged first":      ("#eb6834", [.540, .577, .590, .580, .589, .590, .596]),
    "random (control)":              ("#8a8983", [.540, .546, .551, .553, .580, .586, .596]),
}

fig, ax = plt.subplots(figsize=(6.4, 4.0))
style(ax)
for name, (c, ys) in POLICY.items():
    dashed = name.startswith("random")
    ax.plot(BUDGET, ys, color=c, linewidth=2, label=name, zorder=3,
            linestyle=(0, (4, 2)) if dashed else "-", solid_capstyle="round")
    ax.plot(BUDGET, ys, "o", color=c, ms=6, zorder=4)

ax.axvline(20, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
# Callouts sit in the two empty regions: above the curves on the left,
# and below the control line in the lower middle.
ax.annotate("20% of the corpus crawled\n→ 89% of the benefit",
            xy=(20, 0.5905), xytext=(31, 0.6045), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9,
                            connectionstyle="arc3,rad=0.2"))
ax.annotate("choosing well beats\ncrawling more",
            xy=(15, 0.5665), xytext=(26, 0.5375), fontsize=8.5, color=INK_SOFT,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9,
                            connectionstyle="arc3,rad=-0.25"))
ax.fill_between([0, 10, 20, 30, 50, 70, 100],
                POLICY["targeted by model uncertainty"][1],
                POLICY["random (control)"][1], color="#2a78d6", alpha=0.07, zorder=2)

ax.set_xlim(-3, 103); ax.set_ylim(0.531, 0.612)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("dev mean macro-F1")
ax.set_xlabel("share of the corpus crawled  (collection cost →)", labelpad=8)
ax.set_title("Most of the crawl is optional", loc="left",
             fontsize=11, fontweight="bold", pad=12)
leg = ax.legend(frameon=False, loc="lower right", fontsize=8.5,
                handlelength=2.2, labelspacing=0.5)
for t in leg.get_texts():
    t.set_color(INK)
ax.annotate("ST3 always answered at level 1; ST1/ST2 escalate to the crawled model within the budget.",
            (0, 0), xytext=(0, -46), textcoords="offset points",
            xycoords="axes fraction", fontsize=7.5, color=INK_MUTED)
fig.subplots_adjust(left=0.10, right=0.97, top=0.87, bottom=0.24)
fig.savefig(OUT / "fig2_cascade.pdf")
fig.savefig(OUT / "fig2_cascade.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# The routed system for Section 2. Drawn rather than ASCII because the two things ASCII
# cannot carry are the point: the paths differ in COST, and the cheap one serves
# the sub-task with legal consequence.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.6, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(-3.5, 50); ax.axis("off")

EDGE, FILL = "#d6d5d0", "#f7f7f5"


def box(x, y, w, h, lines, edge=EDGE, fill=FILL, fs=8.4, sub_fs=7.0):
    """lines: list of (text, kind) where kind is 'title' or 'sub'."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.5",
                 linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2))
    heights = [1.0 if k == "title" else 0.78 for _, k in lines]
    cur = y + h / 2 + sum(heights) * 3.0 / 2
    for (txt, kind), hh in zip(lines, heights):
        cur -= hh * 3.0 / 2
        ax.text(x + w / 2, cur, txt, ha="center", va="center",
                fontsize=fs if kind == "title" else sub_fs,
                color=INK if kind == "title" else INK_SOFT,
                fontweight="bold" if kind == "title" else "normal", zorder=3)
        cur -= hh * 3.0 / 2


def arrow(x1, y1, x2, y2, color=INK_MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=8, linewidth=1.0, color=color, zorder=1,
                 shrinkA=0, shrinkB=0))


for hx, ht in [(13, "INPUT"), (41, "MODELS"), (66, "COMBINE"), (88, "OUTPUT")]:
    ax.text(hx, 46.5, " ".join(ht), ha="center", fontsize=6.6,
            color=INK_MUTED, fontweight="bold")

# --- cheap path -> ST3 ------------------------------------------------------
box(1, 30.5, 24, 12.5, [("transcript", "title"), ("level 1 · lowest cost", "sub"),
                        ("scales to every video", "sub")])
box(29, 37.5, 26, 5.5, [("ModernBERT-large  ×6", "title")],
    edge=SERIES["ST3"], fill="#f0faf6", fs=7.8)
box(29, 30.0, 26, 5.5, [("QLoRA Qwen2.5-7B  ×3", "title")],
    edge=SERIES["ST3"], fill="#f0faf6", fs=7.8)
box(58, 30.5, 18, 12.5, [("route by flag", "title"),
                         ("2 disclosure flags:", "sub"), ("both models averaged", "sub"),
                         ("other 6: encoder", "sub")], fs=7.8, sub_fs=6.5)
box(79, 30.5, 20, 12.5, [("ST3", "title"), ("compliance flags", "sub"),
                         ("+ constraint layer", "sub")],
    edge=SERIES["ST3"], fill="#f0faf6")
arrow(25, 40.2, 29, 40.2); arrow(25, 33.3, 29, 32.8)
arrow(55, 40.2, 58, 38.5); arrow(55, 32.8, 58, 35.0)
arrow(76, 36.7, 79, 36.7)

# --- crawled path -> ST1, ST2 ----------------------------------------------
box(1, 6.0, 24, 16.0, [("+ video metadata", "title"), ("+ channel", "title"),
                       ("+ product page", "title"),
                       ("levels 1–4 · one crawl per video", "sub")],
    fs=7.6, sub_fs=6.5)
box(29, 15.0, 26, 5.5, [("ModernBERT-large  ×6", "title")],
    edge="#c9c8c2", fill="#f2f4f8", fs=7.8)
box(29, 7.5, 26, 5.5, [("TF-IDF + logistic reg.", "title")],
    edge=SERIES["ST2"], fill="#fdf0e9", fs=7.8)
box(58, 6.5, 18, 6.0, [("average 50/50", "title")], fs=7.8)
box(79, 15.5, 20, 6.0, [("ST1  commercial type", "title")],
    edge=SERIES["ST1"], fill="#eef4fc", fs=7.8)
box(79, 6.5, 20, 6.0, [("ST2  product category", "title")],
    edge=SERIES["ST2"], fill="#fdf0e9", fs=7.8)
arrow(25, 17.8, 29, 17.8); arrow(25, 10.3, 29, 10.3)
arrow(55, 17.8, 79, 18.5)
arrow(55, 16.2, 58, 10.5); arrow(55, 10.3, 58, 9.0)
arrow(76, 9.5, 79, 9.5)

ax.text(2, 26.0, "level 1 — screen every video", fontsize=7.0,
        color=SERIES["ST3"], fontweight="bold")
ax.text(2, 1.4, "levels 1–4 — crawl, budget-capped (Section 1)", fontsize=7.0,
        color=INK_MUTED, fontweight="bold")
ax.text(50, -1.2,
        "The sub-task with legal consequence runs on the cheapest field; "
        "the crawl is bought only to identify what is being sold.",
        ha="center", fontsize=7.6, color=INK_SOFT, style="italic")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "fig3_architecture.pdf")
fig.savefig(OUT / "fig3_architecture.png", dpi=200)
plt.close(fig)
print("  fig3_architecture written")

# ---------------------------------------------------------------- figure 4
# The Section 4 result. Left: every adoption call we both assessed and measured on the
# evaluation set. Right: the mechanism behind the one that failed -- a rare class
# driven to zero predictions, which fixes its F1 at zero under a macro average.
import numpy as np

BAD = "#c0392b"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                               gridspec_kw={"width_ratios": [1, 1]})

# --- A: what the criterion predicted, and what the test set paid -------------
CALLS = [   # label, evidence, test effect, sub-task, criterion right?
    ("ST2 classical blend",   "adopt · 5/5 folds",  -0.0293, "ST2", False),
    ("zero-shot LLM swap",    "never assessed",     -0.0050, "ST3", None),
    ("ST1 logit adj. \u03c4 = 0.3", "reject · 7/10",  -0.0405, "ST1", True),
    ("no_flag \u2265 0.6",       "adopt · 10/10",      +0.0206, "ST3", True),
    ("ST3 disclosure hybrid", "adopt · 20/20",      +0.0871, "ST3", True),
]
ys = np.arange(len(CALLS)) * 1.0
for y, (lab, ev, eff, st, ok) in zip(ys, CALLS):
    axA.barh(y, eff, height=0.34, color=SERIES[st], edgecolor="white",
             linewidth=1.4, zorder=3)
    axA.text(eff + (0.005 if eff > 0 else -0.005), y, f"{eff:+.3f}", va="center",
             ha="left" if eff > 0 else "right", fontsize=8.4, color=INK,
             fontweight="bold", zorder=4)
    axA.text(-0.155, y + 0.20, lab, va="center", ha="left", fontsize=8.2, color=INK)
    axA.text(-0.155, y - 0.20, ev, va="center", ha="left", fontsize=7.0,
             color=BAD if ok is False else INK_MUTED,
             fontweight="bold" if ok is False else "normal")
axA.axvline(0, color=INK_SOFT, linewidth=0.9, zorder=2)
axA.set_xlim(-0.157, 0.115); axA.set_ylim(-0.85, len(CALLS) - 0.35)
axA.set_yticks([]); axA.set_xlabel("effect on the evaluation set")
axA.set_xticks([-0.05, 0, 0.05, 0.10])
axA.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.2f}"))
for side in ("top", "right", "left"):
    axA.spines[side].set_visible(False)
axA.spines["bottom"].set_bounds(-0.06, 0.11)
axA.spines["bottom"].set_linewidth(0.8)
axA.grid(axis="x", color=GRID, linewidth=0.7, zorder=0); axA.set_axisbelow(True)
axA.tick_params(length=0)
axA.set_title("Four of five adoption calls were correct",
              fontsize=10.6, fontweight="bold", color=INK, loc="left", x=-0.30, pad=14)
axA.annotate("the strongest held-out\nevidence we had — and wrong",
             xy=(-0.020, 0.18), xytext=(0.012, 1.05), fontsize=7.4, color=BAD,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=BAD, linewidth=0.9,
                             connectionstyle="arc3,rad=-0.3"))

# --- B: the mechanism -------------------------------------------------------
CLS = [  # class, train n, encoder preds, blend preds
    ("gambling", 17, 3, 0), ("toys", 85, 37, 28), ("gambling_adjacent", 109, 47, 24),
    ("education", 152, 26, 20), ("financial", 155, 75, 60),
    ("creator_community", 316, 122, 87), ("fashion", 317, 90, 82),
    ("health", 319, 72, 46), ("food", 342, 55, 53), ("other", 499, 147, 101),
    ("hardware_electronics", 636, 106, 92), ("apps", 825, 172, 128),
][::-1]
yb = np.arange(len(CLS))
for y, (name, n, e, b) in zip(yb, CLS):
    zero = (b == 0)
    axB.plot([b, e], [y, y], color=BAD if zero else "#dedcd7",
             linewidth=2.2 if zero else 1.8, zorder=2, solid_capstyle="round")
    axB.scatter([e], [y], s=30, color="#b9b8b3", zorder=3,
                edgecolor="white", linewidth=1.1)
    axB.scatter([b], [y], s=46 if zero else 30, color=BAD if zero else SERIES["ST2"],
                zorder=4, edgecolor="white", linewidth=1.1)
    axB.text(-8, y, f"{name}  ({n})", va="center", ha="right", fontsize=7.4,
             color=BAD if zero else INK_SOFT,
             fontweight="bold" if zero else "normal")
axB.set_xlim(-6, 192); axB.set_ylim(-1.5, len(CLS) - 0.2)
axB.set_yticks([]); axB.set_xticks([0, 50, 100, 150])
axB.set_xlabel("ST2 predictions emitted, 503 evaluation instances")
for side in ("top", "right", "left"):
    axB.spines[side].set_visible(False)
axB.spines["bottom"].set_bounds(0, 175); axB.spines["bottom"].set_linewidth(0.8)
axB.grid(axis="x", color=GRID, linewidth=0.7, zorder=0); axB.set_axisbelow(True)
axB.tick_params(length=0)
axB.set_title("Because 17 training instances sit below the\ninstrument's resolution",
              fontsize=10.6, fontweight="bold", color=INK, loc="left", x=-0.42, pad=14)
axB.scatter([], [], s=30, color="#b9b8b3", label="encoder alone")
axB.scatter([], [], s=30, color=SERIES["ST2"], label="after the 50/50 blend")
axB.legend(loc="lower right", frameon=False, fontsize=7.6, handletextpad=0.35,
           bbox_to_anchor=(1.02, -0.03))
axB.annotate("0 predictions \u2192 F1 = 0.\nOne class of twelve is worth\n0.083 of the macro average.",
             xy=(3, len(CLS) - 1.10), xytext=(72, len(CLS) - 3.6), fontsize=7.4,
             color=BAD, ha="left", va="center",
             arrowprops=dict(arrowstyle="-|>", color=BAD, linewidth=1.0,
                             mutation_scale=8, connectionstyle="arc3,rad=0.3"))
axB.text(-8, -1.15, "training instances in brackets", fontsize=6.8,
         color=INK_MUTED, ha="right")

fig.subplots_adjust(left=0.155, right=0.995, top=0.845, bottom=0.115, wspace=0.60)
fig.savefig(OUT / "fig4_validation.pdf")
fig.savefig(OUT / "fig4_validation.png", dpi=200)
plt.close(fig)
print("  fig4_validation written")

print(f"\nwrote to {OUT}:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
