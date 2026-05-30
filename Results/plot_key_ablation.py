import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

models = [
    ("Vision MobileNetV2",     47.2),
    ("Biosignal SVM",          48.8),
    ("Biosignal RF",           63.1),
    ("Landmark RF",            61.4),
    ("Early Fusion",           64.6),
    ("GNN + Biosig Stacked",   63.1),
    ("DANN Biosignal",         61.6),
    ("EmPath Stacked Fusion",  65.3),
]

models = sorted(models, key=lambda x: x[1])
labels = [m[0] for m in models]
values = [m[1] for m in models]
widths = [v - 40 for v in values]   # bar width = distance from x=40

cmap = plt.cm.RdBu_r
norm = mcolors.Normalize(vmin=45, vmax=68)
bar_colors = [cmap(norm(v)) for v in values]
bar_colors[-1] = cmap(norm(68))  # EmPath → deepest red

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.barh(labels, widths, color=bar_colors, height=0.6, left=40, zorder=3)

for bar, val in zip(bars, values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", ha="left",
            color="#333333", fontsize=9.5, fontweight="bold")

ax.axvline(50, color="#999999", linewidth=1.2, linestyle="--", zorder=2)
ax.text(50.15, -0.7, "Chance", color="#999999", fontsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.025, pad=0.02, aspect=20)
cbar.set_label("Accuracy (%)", fontsize=8, color="#555555")
cbar.ax.tick_params(labelsize=7, colors="#555555")
cbar.outline.set_edgecolor("#cccccc")

ax.set_xlim(40, 70)
ax.set_xlabel("LOSO-67 Accuracy (%)", color="#555555", fontsize=10)
ax.tick_params(axis="y", colors="#333333", labelsize=9.5)
ax.tick_params(axis="x", colors="#888888", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_title("Model Comparison — LOSO-67 Accuracy",
             color="#222222", fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
out = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/key_ablation.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
