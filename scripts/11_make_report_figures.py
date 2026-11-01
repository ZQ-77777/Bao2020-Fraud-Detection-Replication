"""Step 11: build the report figures from the archived summary tables in results_summary/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results_summary"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--", "figure.dpi": 150})
BLUE, RED, GREEN, VIOLET, GREY, AMBER = "#1F4E8C", "#B42318", "#2C7A4B", "#5B47B8", "#8A949E", "#A15C00"

# Figure 1: headline -- frauds caught in the top 1%, 2003-2008
rows = [("Published (Bao et al. 2020)", 16, None, None, BLUE),
        ("Walker (2021): re-run of authors' code", 15, None, None, BLUE),
        ("Walker (2021): original AAER ids", 9, None, None, AMBER),
        ("Ours: no serial-fraud handling, deep trees", 19.33, 18, 22, RED),
        ("Ours: no serial-fraud handling", 13.67, 13, 14, RED),
        ("Ours: authors' original design", 9.00, 7, 10, VIOLET),
        ("Ours: corrected design", 8.00, 7, 9, GREEN),
        ("Ours: logit, 14 ratios (common sample)", 9.00, None, None, GREY)]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.arange(len(rows))[::-1]
for yi, (lab, v, lo, hi, c) in zip(y, rows):
    ax.barh(yi, v, color=c, alpha=0.85)
    if lo is not None:
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color="black", capsize=3, lw=1)
    ax.text((hi if hi is not None else v) + 0.4, yi, f"{v:.1f}" if v % 1 else f"{int(v)}", va="center", fontsize=9)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlabel("Frauds caught in the top 1% of firms (2003-2008 total)")
ax.set_xlim(0, 24)
fig.tight_layout(); fig.savefig(FIG / "fig1_headline_hits.png"); plt.close(fig)

# Figure 3: serial-fraud schemes x model capacity
d = pd.read_csv(SUM / "serial_capacity.csv")
schemes = ["none", "recode_new_p_aaer", "recode_p_aaer", "drop_p_aaer", "drop_firm"]
labels = ["None", "Recode\n(new_p_aaer)", "Recode\n(p_aaer)", "Drop\n(p_aaer)", "Drop\n(firm)"]
settings = [("shallow_3000", "11 leaves, 3000 trees", BLUE), ("deep_3000", "unlimited leaves, 3000 trees", RED),
            ("shallow_400", "11 leaves, 400 trees", GREY)]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = np.arange(len(schemes)); w = 0.26
for i, (key, lab, c) in enumerate(settings):
    s = d[d.setting == key].set_index("scheme").loc[schemes]
    ax.bar(x + (i - 1) * w, s.hits_mean, w, color=c, alpha=0.85, label=lab)
    ax.errorbar(x + (i - 1) * w, s.hits_mean, yerr=[s.hits_mean - s.hits_min, s.hits_max - s.hits_mean],
                fmt="none", color="black", capsize=2, lw=0.8)
ax.axhline(16, color=BLUE, ls=":", lw=1); ax.text(1.0, 16.4, "published: 16", ha="left", fontsize=8, color=BLUE)
ax.set_xticks(x, labels); ax.set_ylabel("Top-1% hits (mean, seed range)")
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.93))
fig.tight_layout(); fig.savefig(FIG / "fig3_serial_capacity.png"); plt.close(fig)

# Figure 2: published vs our scores by year
p = pd.read_csv(SUM / "published_compare.csv")
fig, ax = plt.subplots(figsize=(7.5, 3.2))
x = np.arange(len(p)); w = 0.38
ax.bar(x - w / 2, p.published_hits, w, color=BLUE, label="Authors' published scores")
ax.bar(x + w / 2, p.our_hits, w, color=VIOLET, label="Our scores (original design, 1 seed)")
for xi, (sp, ov) in enumerate(zip(p.spearman, p.top1pct_overlap)):
    ax.text(xi, 8.6, f"rho={sp:.2f}\noverlap={ov:.0%}", ha="center", fontsize=7, color=GREY)
ax.set_xticks(x, p.year); ax.set_ylim(0, 10); ax.set_ylabel("Top-1% hits")
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.78))
fig.tight_layout(); fig.savefig(FIG / "fig2_published_vs_ours.png"); plt.close(fig)

# Figure 4: case-level decomposition -- capture rate by category
lk = pd.read_csv(SUM / "leak_categories.csv").set_index("source")
cats = ["split_leak", "same_case", "new"]
base = lk.loc["all_test_year_frauds", cats]
src = [("published", "Authors' published scores", BLUE), ("ours_recode_new_p_aaer", "Ours: original design", VIOLET),
       ("ours_recode_p_aaer", "Ours: corrected design", GREEN)]
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
x = np.arange(3); w = 0.26
for i, (key, lab, c) in enumerate(src):
    axes[0].bar(x + (i - 1) * w, lk.loc[key, cats], w, color=c, label=lab)
    axes[1].bar(x + (i - 1) * w, 100 * lk.loc[key, cats] / base, w, color=c)
names = [f"Split-id leak\n(n={int(base.split_leak)})", f"Same case, recoded\n(n={int(base.same_case)})",
         f"New fraud\n(n={int(base.new)})"]
axes[0].set_xticks(x, names, fontsize=8); axes[0].set_ylabel("Top-1% hits"); axes[0].set_title("Hits by category", fontsize=10)
axes[1].set_xticks(x, names, fontsize=8); axes[1].set_ylabel("Capture rate (%)"); axes[1].set_title("Hits / frauds in category", fontsize=10)
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig4_leak_categories.png"); plt.close(fig)

# Figure 6: tuning, leaky vs corrected
t = pd.read_csv(SUM / "tuning.csv")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
idx = np.arange(len(t))
axes[0].plot(idx, t.valid_auc_leaky, "o--", color=GREY, ms=3, label="Tuning without serial-fraud handling")
axes[0].plot(idx, t.valid_auc_corrected, "o-", color=BLUE, ms=3, label="Corrected")
axes[0].set_ylabel("Validation AUC (2001)"); axes[0].legend(frameon=False, fontsize=7.5)
axes[1].plot(idx, t.valid_hits_corrected, "o-", color=GREEN, ms=3)
axes[1].set_ylabel("Validation top-1% hits (corrected)")
for ax in axes:
    ax.set_xticks(idx[::2], t.n_trees[::2], fontsize=8); ax.set_xlabel("Number of trees")
fig.tight_layout(); fig.savefig(FIG / "fig6_tuning.png"); plt.close(fig)

# Figure 5: common-sample model comparison
c = pd.read_csv(SUM / "common_sample.csv")
fig, ax = plt.subplots(figsize=(7.5, 3.2))
y = np.arange(len(c))[::-1]
colors = [GREY if m.startswith("Logit") else (GREEN if m.startswith("RUS") else BLUE) for m in c.model]
ax.barh(y, c.hits_mean, color=colors, alpha=0.85)
ax.errorbar(c.hits_mean, y, xerr=[c.hits_mean - c.hits_min, c.hits_max - c.hits_mean], fmt="none", color="black", capsize=2, lw=0.8)
for yi, (h, a) in zip(y, zip(c.hits_mean, c.auc)):
    ax.text(10.4, yi, f"AUC {a:.3f}", va="center", fontsize=8, color=GREY)
ax.set_yticks(y, c.model); ax.set_xlim(0, 12); ax.set_xlabel("Top-1% hits, common sample (126,483 firm-years)")
fig.tight_layout(); fig.savefig(FIG / "fig5_common_sample.png"); plt.close(fig)
print("Figures written to", FIG)
