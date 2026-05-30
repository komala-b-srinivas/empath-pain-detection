# EmPath: Multimodal Pain Intensity Detection

Subject-independent pain classification from biosignals and facial landmarks using stacked generalization — BioVid Heat Pain Database, LOSO-67 evaluation.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://komala-b-srinivas-empath-app-oxt9of.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ACM%20SAC%202027-blue)](paper/empath_acm_sac.tex)

**Live demo:** https://komala-b-srinivas-empath-app-oxt9of.streamlit.app/

**Authors:** Komala Belur Srinivas · Dr. Purna Prasad — Hofstra University (2026)

---

## Overview

EmPath discriminates between two adjacent pain intensity levels (PA2 ~43C and PA3 ~44C) from the BioVid Heat Pain Database using:
- **35 biosignal features** from GSR, ECG, EMG, and HRV (NeuroKit2)
- **22 facial landmark features** from MediaPipe FaceMesh (468-point mesh, 24 frames per trial)
- **Stacked generalization**: RF(biosignal) + RF(landmark) -> Logistic Regression meta-learner
- **LOSO evaluation** on 67 thermally reactive subjects (20 non-reactive excluded)

---

## Results

### Primary Result

| Model | Eval Protocol | Accuracy | SD | AUC | F1 |
|-------|--------------|----------|-----|-----|-----|
| Chance baseline | - | 50.0% | - | 0.500 | 0.500 |
| Biosignal RF (no norm) | LOSO-67 | 59.9% | 13.2% | - | - |
| Biosignal RF | LOSO-67 | 63.1% | 11.6% | 0.668 | 0.631 |
| Landmark RF | LOSO-67 | 61.4% | 13.1% | 0.641 | 0.614 |
| Early fusion (concat) | LOSO-67 | 64.6% | 13.8% | 0.701 | 0.646 |
| **EmPath Stacked Fusion** | **LOSO-67** | **65.3%** | **14.1%** | **0.719** | **0.653** |

### Full Ablation (26 variants)

| Model | Eval Protocol | Accuracy |
|-------|--------------|----------|
| Vision MobileNetV2 | Random split | 47.2% |
| Biosignal SVM | Random split | 48.8% |
| Biosignal MLP | Random split | 51.2% |
| Landmark RF (flat, split) | Random split | 51.6% |
| Biosignal XGBoost | Random split | 54.1% |
| Biosignal TCN | Random split | 55.9% |
| Biosignal RF (50 subj) | Random split | 56.9% |
| Biosignal RF (67 subj) | Random split | 59.5% |
| PainFormer | LOSO-67 | 53.1% |
| BIOT | LOSO-67 | 54.4% |
| BIOT + hand features | LOSO-67 | 60.8% |
| Tiny-BioMoE | LOSO-67 | 56.7% |
| Tiny-BioMoE + hand feats | LOSO-67 | 61.7% |
| Biosignal RF | LOSO-67 | 63.1% |
| Landmark RF (flat) | LOSO-67 | 61.4% |
| Hybrid CNN + hand feats | LOSO-67 | 59.7% |
| Attention fusion | LOSO-67 | 61.1% |
| Ordinal MLP | LOSO-67 | 61.2% |
| Early fusion (concat RF) | LOSO-67 | 64.6% |
| Subject adaptation RF | LOSO-67 | 65.1% |
| CORAL ordinal MLP | LOSO-67 | 65.3% |
| **EmPath Stacked Fusion** | **LOSO-67** | **65.3% +/- 14.1%** |
| GNN landmarks only | LOSO-67 | 51.7% +/- 9.8% |
| GNN + biosignal stacked | LOSO-67 | 63.1% +/- 11.9% |
| DANN biosignal | LOSO-67 | 61.6% +/- 10.3% |
| DANN + RF landmarks | LOSO-67 | 64.7% +/- 11.8% |
| CrossMod cross-attention | LOSO-67 | 63.1% +/- 11.1% |
| Velocity RF only | LOSO-67 | 60.0% +/- 11.9% |
| Velocity + biosig stacked | LOSO-67 | 64.0% +/- 12.7% |

---

## System Architecture

```
BioVid Database
    |                          |
MP4 Video Files          Biosignal TSVs
    |                          |
Face Extraction          Signal Windowing
    |                          |
MediaPipe FaceMesh       NeuroKit2 + Stats
    |                          |
22 Landmark Features     35 Biosignal Features
    \                         /
      Person-Specific z-Score
      /                     \
RF Landmark             RF Biosignal
(p_lm [2])              (p_bio [2])
      \                     /
    [p_bio || p_lm]  (4-dim)
            |
   LogReg Meta-Learner
            |
      PA2 / PA3 Prediction
```

**Key design choices:**
- Person-specific z-score normalization: +3.2 pp improvement over global normalization
- Stacked generalization over early fusion: preserves modality structure, avoids scale mismatch
- LOSO cross-validation: no subject identity leakage, strict cross-subject evaluation
- 67 reactive subjects only: 20 non-reactive excluded per BioVid methodology

---

## Repository Structure

```
EmPath_v2/
|
|-- app.py                          Streamlit demo app (live at the link above)
|-- build_ppt.py                    Script to generate the project presentation
|-- requirements.txt                Pinned dependencies
|-- CLAUDE.md                       Project context and experiment log
|-- EmPath_Pipeline_Guide.md        Step-by-step pipeline walkthrough
|
|-- SRC/
|   |-- preprocessing/              All experiment and feature extraction scripts
|   |   |-- extract_biosignals_all87.py      Extract 35 biosignal features
|   |   |-- extract_landmarks_all67.py       Extract 22 facial landmark features
|   |   |-- extract_landmarks_raw_coords.py  Extract raw 468-point coords for GNN
|   |   |-- extract_faces.py / extract_faces_all67.py  Video face extraction
|   |   |-- evaluate_stacked_fusion_loso.py  BASELINE: 65.3% EmPath stacked fusion
|   |   |-- evaluate_landmarks_loso.py       Landmark RF baseline
|   |   |-- evaluate_mlp_loso.py             MLP deep learning baseline
|   |   |-- evaluate_fusion_loso.py          Early fusion baseline
|   |   |-- evaluate_ensemble_loso.py        Ensemble variants
|   |   |-- evaluate_gnn_landmarks_loso.py   GAT on landmark graph
|   |   |-- evaluate_dann_loso.py            DANN adversarial adaptation (61.6%)
|   |   |-- evaluate_velocity_loso.py        Velocity features (60.0%)
|   |   |-- evaluate_crossmod_loso.py        CrossMod attention fusion (63.1%)
|   |   |-- error_analysis_loso.py           SHAP, confusion matrix, per-subject breakdown
|   |   |-- shap_analysis_loso.py            SHAP TreeExplainer standalone
|   |   |-- save_final_model.py              Save production model to Models/
|   |   |-- save_signals_plot.py             Pre-render signal plots for Streamlit Cloud
|   |   `-- prepare_splits.py               Train/val/test split preparation
|   |
|   |-- notebooks/
|   |   |-- EmPath_Training.ipynb            Training notebook (Colab/Kaggle compatible)
|   |   `-- colab_gnn_experiment.ipynb       GNN experiment notebook for Colab
|   |
|   `-- utils/
|       `-- run_shap.slurm                   SLURM job script for HPC SHAP runs
|
|-- Data/
|   |-- Raw/                                 BioVid raw data (NOT in git, too large)
|   |   |-- video/                           MP4 files by subject_name/
|   |   |-- biosignals_filtered/             TSV files with gsr/ecg/emg columns
|   |   `-- starting_point/samples.csv       Master sample list
|   `-- splits/                              Pre-computed train/val/test CSVs
|
|-- Models/
|   |-- empath_model.pkl                     Saved stacked fusion model (sklearn 1.7.2)
|   `-- demo_samples.csv                     2680-sample feature table for Streamlit
|
|-- Results/
|   |-- biosignals_hrv/all_67_hrv.csv        35 biosignal features, 2680 samples
|   |-- landmarks_all67/landmarks_all67.csv  22 facial landmark features
|   |-- landmarks_gnn/raw_coords.npz         Raw (N, 24, 468, 2) coords for GNN
|   |-- error_analysis_v2/
|   |   |-- shap_biosignal_ranked.csv        SHAP values (biosignal RF)
|   |   |-- shap_landmark_ranked.csv         SHAP values (landmark RF)
|   |   `-- per_subject_accuracy.csv         67-fold accuracy breakdown
|   |-- signal_plots/                        Pre-rendered PNG plots for Streamlit Cloud
|   `-- showcase.html                        Interactive results visualization
|
`-- rep/
    |-- empath_report.tex                    LaTeX source (journal-style, two-column)
    |-- references.bib                       BibTeX references (45 entries)
    `-- README_latex.md                      Overleaf compilation instructions
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/komalabelursrinivas/empath-pain-detection.git
cd EmPath_v2

# Create environment
conda create -n empath python=3.10 -y
conda activate empath

# Install dependencies
pip install -r requirements.txt

# For GNN experiments only
pip install torch-geometric
```

**Note:** Raw BioVid data is not included. Request access from the original authors (Walter et al., 2013). Place files in Data/Raw/ following the structure above.

---

## How to Run

### Feature Extraction (run once, outputs already in Results/)

```bash
# Biosignal features (35 features, 2680 samples)
python SRC/preprocessing/extract_biosignals_all87.py

# Facial landmark features (22 features)
python SRC/preprocessing/extract_landmarks_all67.py

# Raw landmark coordinates for GNN
python SRC/preprocessing/extract_landmarks_raw_coords.py
```

### Baseline Evaluation

```bash
# EmPath stacked fusion (65.3% +/- 14.1%) - main result
python SRC/preprocessing/evaluate_stacked_fusion_loso.py

# Error analysis: SHAP, confusion matrix, per-subject breakdown
python SRC/preprocessing/error_analysis_loso.py
```

### Advanced Architectures

```bash
# GNN on raw landmark coordinates (51.7%)
python SRC/preprocessing/evaluate_gnn_landmarks_loso.py

# DANN adversarial adaptation (61.6% / 64.7% stacked)
python SRC/preprocessing/evaluate_dann_loso.py

# Velocity features (60.0% / 64.0% stacked)
python SRC/preprocessing/evaluate_velocity_loso.py

# CrossMod cross-attention (63.1%)
python SRC/preprocessing/evaluate_crossmod_loso.py
```

### Streamlit Demo

```bash
streamlit run app.py
```

---

## Key Findings

1. **Multimodal fusion consistently outperforms unimodal baselines.** Stacked generalization (65.3%) beats biosignal-only RF (63.1%) and landmark-only RF (61.4%), and also beats early concatenation fusion (64.6%). Preserving modality boundaries and combining calibrated probability estimates is more effective than mixing heterogeneous raw features.

2. **Person-specific normalization is the single most impactful preprocessing decision.** Normalizing each subject's features to their own mean/std yields a +3.2 pp improvement (59.9% to 63.1% for biosignal RF). This approximates the effect of individual calibration without requiring labelled test data.

3. **GSR slope and facial movement variability are the dominant discriminants.** SHAP analysis identifies `gsr_slope` (mean |SHAP| = 0.0372) and `mouth_height_std` (mean |SHAP| = 0.0289) as the top features. Temporal variability features consistently outrank mean-configuration features for facial landmarks.

4. **The performance ceiling for this protocol is approximately 65%.** 26 architectural variants including deep learning (TCN, MLP, R3D-18), foundation models (BIOT, PainFormer, Tiny-BioMoE), GNNs, DANN, and attention fusion all converge near or below 65.3%. The ceiling is set by inter-individual physiological variability and the 1-degree stimulus difference, not model capacity.

---

## Publication

This work is being prepared for submission to **ACM SAC 2027** (Health Informatics / BCB track).
The LaTeX source is at [`paper/empath_acm_sac.tex`](paper/empath_acm_sac.tex).

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{beluursrinivas2027empath,
  author    = {Belur Srinivas, Komala and Prasad, Purna},
  title     = {{EmPath}: Multimodal Pain Intensity Detection via Stacked Generalization
               of Biosignals and Facial Landmarks},
  booktitle = {Proceedings of the 42nd ACM/SIGAPP Symposium on Applied Computing (SAC)},
  year      = {2027},
  publisher = {ACM},
  note      = {Under review}
}
```

---

## License

MIT License. See LICENSE for details.

The BioVid Heat Pain Database is subject to its own data access agreement. Contact the original authors for access.
