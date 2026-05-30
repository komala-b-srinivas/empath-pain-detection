"""
t-SNE visualization of the 4-dim stacker meta-features.

Collects [P(PA2|bio), P(PA3|bio), P(PA2|lm), P(PA3|lm)] for every
test sample across all 67 LOSO folds, then plots three views:
  1. Colored by true label  (PA2 vs PA3)
  2. Colored by subject ID  (shows cross-subject spread)
  3. Colored by correct / incorrect prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# ── Paths ─────────────────────────────────────────────────────────────────────
LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/landmarks_all67/landmarks_all67.csv"
BIOSIG_CSV    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/biosignals_hrv/all_67_hrv.csv"
OUT_DIR       = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/tsne"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & merge ──────────────────────────────────────────────────────────────
print("Loading data...")
lm_df  = pd.read_csv(LANDMARKS_CSV)
bio_df = pd.read_csv(BIOSIG_CSV)

EXCLUDE  = ["subject_id", "sample_name", "class_name", "label"]
lm_cols  = [c for c in lm_df.columns  if c not in EXCLUDE]
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]

bio_dict = {row["sample_name"]: row for _, row in bio_df.iterrows()}
rows = []
for _, lm_row in lm_df.iterrows():
    sname = lm_row["sample_name"]
    if sname not in bio_dict:
        continue
    bio_row = bio_dict[sname]
    rows.append({
        "subject_id": lm_row["subject_id"],
        "label":      lm_row["label"],
        "bio":        np.nan_to_num(bio_row[bio_cols].values.astype(float)),
        "lm":         np.nan_to_num(lm_row[lm_cols].values.astype(float)),
    })

X_bio  = np.array([r["bio"]       for r in rows])
X_lm   = np.array([r["lm"]        for r in rows])
y      = np.array([r["label"]      for r in rows])
groups = np.array([r["subject_id"] for r in rows])
print(f"Matched: {len(rows)} samples, {len(np.unique(groups))} subjects")

# ── Normalization ─────────────────────────────────────────────────────────────
def person_norm_train(X, g):
    X_norm = X.copy()
    for sid in np.unique(g):
        mask = g == sid
        mean = X[mask].mean(axis=0)
        std  = X[mask].std(axis=0); std[std == 0] = 1
        X_norm[mask] = (X[mask] - mean) / std
    return X_norm

def person_norm_test(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0); std[std == 0] = 1
    return (X - mean) / std

RF_PARAMS = dict(n_estimators=300, max_depth=4,
                 min_samples_split=10, max_features="sqrt",
                 random_state=42, n_jobs=-1)

# ── Collect meta-features across all LOSO folds ───────────────────────────────
print("\nCollecting meta-features (67 LOSO folds)...")
logo = LeaveOneGroupOut()

meta_all   = np.zeros((len(rows), 4))   # [P(PA2|bio), P(PA3|bio), P(PA2|lm), P(PA3|lm)]
y_pred_all = np.zeros(len(rows), dtype=int)
order      = np.zeros(len(rows), dtype=int)   # original sample index

for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_bio, y, groups), start=1):

    g_tr = groups[train_idx]
    y_tr = y[train_idx]

    Xb_tr = person_norm_train(X_bio[train_idx], g_tr)
    Xb_te = person_norm_test(X_bio[test_idx])
    Xl_tr = person_norm_train(X_lm[train_idx],  g_tr)
    Xl_te = person_norm_test(X_lm[test_idx])

    rf_bio = RandomForestClassifier(**RF_PARAMS).fit(Xb_tr, y_tr)
    rf_lm  = RandomForestClassifier(**RF_PARAMS).fit(Xl_tr, y_tr)

    bio_te_p = rf_bio.predict_proba(Xb_te)
    lm_te_p  = rf_lm.predict_proba(Xl_te)
    X_meta_te = np.hstack([bio_te_p, lm_te_p])

    X_meta_tr = np.hstack([rf_bio.predict_proba(Xb_tr),
                            rf_lm.predict_proba(Xl_tr)])
    meta_lr = LogisticRegression(max_iter=1000, random_state=42)
    meta_lr.fit(X_meta_tr, y_tr)

    meta_all[test_idx]   = X_meta_te
    y_pred_all[test_idx] = meta_lr.predict(X_meta_te)

    if fold % 20 == 0:
        print(f"  fold {fold}/67")

acc = accuracy_score(y, y_pred_all)
print(f"\nOverall LOSO accuracy: {acc*100:.1f}%")

# ── t-SNE on 4-dim meta-features ──────────────────────────────────────────────
print("Running t-SNE (perplexity=40)...")
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
            random_state=42, init="pca", learning_rate="auto")
Z = tsne.fit_transform(meta_all)   # (2680, 2)

correct = (y_pred_all == y).astype(int)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor("#0F1923")
for ax in axes:
    ax.set_facecolor("#0F1923")
    ax.tick_params(colors="#8899AA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3448")

S = 8    # marker size
A = 0.65 # alpha

# ── Panel 1: True label ───────────────────────────────────────────────────────
ax = axes[0]
colors = np.where(y == 0, "#38BDF8", "#F472B6")   # PA2=sky, PA3=pink
ax.scatter(Z[y==0, 0], Z[y==0, 1], c="#38BDF8", s=S, alpha=A, label="PA2 (~43°C)")
ax.scatter(Z[y==1, 0], Z[y==1, 1], c="#F472B6", s=S, alpha=A, label="PA3 (~45°C)")
ax.set_title("True Label", color="white", fontsize=13, fontweight="bold", pad=10)
ax.legend(handles=[
    mpatches.Patch(color="#38BDF8", label="PA2 (~43°C)"),
    mpatches.Patch(color="#F472B6", label="PA3 (~45°C)"),
], facecolor="#1E3448", edgecolor="#1E3448", labelcolor="white", fontsize=9)
ax.set_xlabel("t-SNE 1", color="#8899AA", fontsize=9)
ax.set_ylabel("t-SNE 2", color="#8899AA", fontsize=9)

# ── Panel 2: Subject ID ───────────────────────────────────────────────────────
ax = axes[1]
unique_subjects = np.unique(groups)
cmap = plt.get_cmap("tab20")
for i, sid in enumerate(unique_subjects):
    mask = groups == sid
    color = cmap(i % 20)
    ax.scatter(Z[mask, 0], Z[mask, 1], c=[color], s=S, alpha=0.55)
ax.set_title("Subject Identity (67 subjects)", color="white",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("t-SNE 1", color="#8899AA", fontsize=9)
ax.set_ylabel("t-SNE 2", color="#8899AA", fontsize=9)

# small note
ax.text(0.02, 0.02, "Each colour = one subject", transform=ax.transAxes,
        color="#8899AA", fontsize=8)

# ── Panel 3: Correct vs incorrect ────────────────────────────────────────────
ax = axes[2]
ax.scatter(Z[correct==1, 0], Z[correct==1, 1],
           c="#34D399", s=S, alpha=A, label=f"Correct ({correct.sum()})")
ax.scatter(Z[correct==0, 0], Z[correct==0, 1],
           c="#F87171", s=S, alpha=0.85, label=f"Wrong ({(1-correct).sum()})")
ax.set_title("Prediction Outcome", color="white", fontsize=13,
             fontweight="bold", pad=10)
ax.legend(handles=[
    mpatches.Patch(color="#34D399", label=f"Correct  ({correct.sum()})"),
    mpatches.Patch(color="#F87171", label=f"Wrong  ({(1-correct).sum()})"),
], facecolor="#1E3448", edgecolor="#1E3448", labelcolor="white", fontsize=9)
ax.set_xlabel("t-SNE 1", color="#8899AA", fontsize=9)
ax.set_ylabel("t-SNE 2", color="#8899AA", fontsize=9)

# ── Shared title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "t-SNE of Stacker Meta-Features  [P(PA2|bio), P(PA3|bio), P(PA2|lm), P(PA3|lm)]",
    color="white", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "tsne_meta_features.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
plt.show()
