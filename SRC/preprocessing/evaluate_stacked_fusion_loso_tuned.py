"""
HP tuning for EmPath stacked fusion — fast two-phase approach.

Phase 1 : GroupKFold(10) on all 67 subjects → find best RF + LogReg HPs
          (~5-10 min)
Phase 2 : Full 67-fold LOSO with those fixed best HPs
          (~3-5 min)

Why not nested-per-fold: nested LOSO would take ~22 hrs on CPU.
GroupKFold(10) uses all subjects for tuning while still respecting
subject identity (no subject appears in both train and val per fold).
"""

import os, json, time, collections
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GroupKFold, LeaveOneGroupOut,
    RandomizedSearchCV, GridSearchCV,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/landmarks_all67/landmarks_all67.csv"
BIOSIG_CSV    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/biosignals_hrv/all_67_hrv.csv"
OUT_DIR       = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2/Results/hp_tuning"
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
print(f"Matched: {len(rows)} samples, {len(np.unique(groups))} subjects\n")

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

# ── HP search spaces ──────────────────────────────────────────────────────────
RF_PARAM_DIST = {
    "n_estimators":      randint(100, 401),       # 100–400
    "max_depth":         [3, 4, 5, 6, None],
    "min_samples_leaf":  randint(1, 11),           # 1–10
    "min_samples_split": [5, 10, 15, 20],
    "max_features":      ["sqrt", "log2", 0.5],
}
LR_PARAM_GRID = {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: One-time HP search via GroupKFold(10)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1 — HP Search (GroupKFold-10 on all 67 subjects)")
print("=" * 60)

gkf  = GroupKFold(n_splits=10)
X_bio_norm = person_norm_train(X_bio, groups)
X_lm_norm  = person_norm_train(X_lm,  groups)

t0 = time.time()

# ── Tune RF_bio ───────────────────────────────────────────────────────────────
print("\nTuning RF_bio  (n_iter=30, cv=GroupKFold(10)) ...")
rs_bio = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),
    RF_PARAM_DIST, n_iter=30,
    cv=gkf, scoring="accuracy",
    n_jobs=-1, random_state=42, refit=True, verbose=1,
)
rs_bio.fit(X_bio_norm, y, groups=groups)
print(f"  Best RF_bio params : {rs_bio.best_params_}")
print(f"  Best CV accuracy   : {rs_bio.best_score_*100:.1f}%")

# ── Tune RF_lm ────────────────────────────────────────────────────────────────
print("\nTuning RF_lm   (n_iter=30, cv=GroupKFold(10)) ...")
rs_lm = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),
    RF_PARAM_DIST, n_iter=30,
    cv=gkf, scoring="accuracy",
    n_jobs=-1, random_state=42, refit=True, verbose=1,
)
rs_lm.fit(X_lm_norm, y, groups=groups)
print(f"  Best RF_lm  params : {rs_lm.best_params_}")
print(f"  Best CV accuracy   : {rs_lm.best_score_*100:.1f}%")

# ── Tune LogReg C on meta-features ────────────────────────────────────────────
print("\nTuning LogReg C on meta-features (GroupKFold(10)) ...")
bio_probs = rs_bio.best_estimator_.predict_proba(X_bio_norm)
lm_probs  = rs_lm.best_estimator_.predict_proba(X_lm_norm)
X_meta    = np.hstack([bio_probs, lm_probs])

gs_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    LR_PARAM_GRID,
    cv=gkf, scoring="accuracy",
    n_jobs=-1, verbose=1,
)
gs_lr.fit(X_meta, y, groups=groups)
best_C = gs_lr.best_params_["C"]
print(f"  Best LogReg C      : {best_C}")
print(f"  Best CV accuracy   : {gs_lr.best_score_*100:.1f}%")

phase1_time = time.time() - t0
print(f"\nPhase 1 done in {phase1_time/60:.1f} min")

best_rf_bio_params = rs_bio.best_params_
best_rf_lm_params  = rs_lm.best_params_

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Full LOSO with tuned HPs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2 — Full LOSO (67 folds) with tuned HPs")
print("=" * 60)

logo = LeaveOneGroupOut()
accs_tuned    = []
accs_baseline = []
y_true_all, y_pred_all = [], []
t1 = time.time()

for fold, (train_idx, test_idx) in enumerate(
        logo.split(X_bio, y, groups), start=1):

    g_tr = groups[train_idx]
    y_tr = y[train_idx];  y_te = y[test_idx]

    Xb_tr = person_norm_train(X_bio[train_idx], g_tr)
    Xb_te = person_norm_test(X_bio[test_idx])
    Xl_tr = person_norm_train(X_lm[train_idx],  g_tr)
    Xl_te = person_norm_test(X_lm[test_idx])

    # ── Tuned models ──────────────────────────────────────────────────────────
    rf_bio_t = RandomForestClassifier(**best_rf_bio_params, random_state=42, n_jobs=-1)
    rf_lm_t  = RandomForestClassifier(**best_rf_lm_params,  random_state=42, n_jobs=-1)
    rf_bio_t.fit(Xb_tr, y_tr);  rf_lm_t.fit(Xl_tr, y_tr)

    X_meta_tr = np.hstack([rf_bio_t.predict_proba(Xb_tr),
                            rf_lm_t.predict_proba(Xl_tr)])
    X_meta_te = np.hstack([rf_bio_t.predict_proba(Xb_te),
                            rf_lm_t.predict_proba(Xl_te)])

    meta_t = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    meta_t.fit(X_meta_tr, y_tr)
    y_pred = meta_t.predict(X_meta_te)
    accs_tuned.append(accuracy_score(y_te, y_pred))
    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(y_pred.tolist())

    # ── Baseline models (fixed HPs) ───────────────────────────────────────────
    rf_bio_b = RandomForestClassifier(n_estimators=300, max_depth=4,
                                      min_samples_split=10, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_lm_b  = RandomForestClassifier(n_estimators=300, max_depth=4,
                                      min_samples_split=10, max_features="sqrt",
                                      random_state=42, n_jobs=-1)
    rf_bio_b.fit(Xb_tr, y_tr);  rf_lm_b.fit(Xl_tr, y_tr)

    X_meta_b_tr = np.hstack([rf_bio_b.predict_proba(Xb_tr),
                              rf_lm_b.predict_proba(Xl_tr)])
    X_meta_b_te = np.hstack([rf_bio_b.predict_proba(Xb_te),
                              rf_lm_b.predict_proba(Xl_te)])
    meta_b = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_b.fit(X_meta_b_tr, y_tr)
    accs_baseline.append(accuracy_score(y_te, meta_b.predict(X_meta_b_te)))

    if fold % 10 == 0 or fold == 1:
        print(f"  Fold {fold:2d}/67 | tuned={np.mean(accs_tuned)*100:.1f}% | "
              f"baseline={np.mean(accs_baseline)*100:.1f}% | "
              f"elapsed={( time.time()-t1)/60:.1f}m")

phase2_time = time.time() - t1

# ── Final results ──────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  RESULTS")
print(f"{'='*62}")
print(f"  Baseline (fixed HPs)  : {np.mean(accs_baseline)*100:.1f}% ± {np.std(accs_baseline)*100:.1f}%")
print(f"  Tuned (HP search)     : {np.mean(accs_tuned)*100:.1f}%    ± {np.std(accs_tuned)*100:.1f}%")
delta = (np.mean(accs_tuned) - np.mean(accs_baseline)) * 100
print(f"  Delta                 : {delta:+.1f} pp")
print(f"{'='*62}")
print(f"\nClassification report (tuned):")
print(classification_report(y_true_all, y_pred_all, target_names=["PA2","PA3"]))
print(f"\nBest HPs found:")
print(f"  RF_bio : {best_rf_bio_params}")
print(f"  RF_lm  : {best_rf_lm_params}")
print(f"  LogReg C: {best_C}")
print(f"\nPhase 1 (search): {phase1_time/60:.1f} min")
print(f"Phase 2 (LOSO)  : {phase2_time/60:.1f} min")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "baseline_mean": round(float(np.mean(accs_baseline)), 4),
    "baseline_std":  round(float(np.std(accs_baseline)),  4),
    "tuned_mean":    round(float(np.mean(accs_tuned)), 4),
    "tuned_std":     round(float(np.std(accs_tuned)),  4),
    "delta_pp":      round(delta, 2),
    "best_rf_bio":   {k: (int(v) if hasattr(v, 'item') else v)
                      for k, v in best_rf_bio_params.items()},
    "best_rf_lm":    {k: (int(v) if hasattr(v, 'item') else v)
                      for k, v in best_rf_lm_params.items()},
    "best_C":        float(best_C),
    "per_fold_tuned":    [round(float(a),4) for a in accs_tuned],
    "per_fold_baseline": [round(float(a),4) for a in accs_baseline],
}
out_path = os.path.join(OUT_DIR, "tuning_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_path}")
