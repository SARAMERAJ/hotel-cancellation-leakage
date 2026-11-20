# Duplicate Leakage, Temporal Drift and Model Choice in Hotel Booking Cancellation Prediction

Code for the paper *Duplicate Leakage, Temporal Drift and Model Choice in Hotel Booking Cancellation Prediction: A Leakage-Controlled Comparison of Deep Learning and Gradient Boosting* (IEEE Access, revised version).

All notebooks are saved with their outputs, so the results can be read without running anything. Table and section numbers below refer to the revised manuscript.

## Original analyses (notebooks)

| Notebook | What it reproduces |
| --- | --- |
| `Notebook_A_CV_Tuning_SHAP.ipynb` | Duplicate removal, leakage-controlled preprocessing, stratified 5-fold cross-validation of logistic regression, 1D CNN, DNN, CNN+DNN hybrid, XGBoost and LightGBM with hyperparameter search inside each training fold, Friedman and McNemar tests, the original raw-data 80/20 random-split comparison (Section IV-E), and the SHAP analysis (Tables II, III, V and VI; Figs. 1–3 and 5–7) |
| `Notebook_B_Time_Ordered.ipynb` | Expanding-window evaluation on the three original future periods, with hyperparameter search inside each window (Tables IV and IX) |
| `Notebook_C_Parking_Check.ipynb` | The first submission's single-model parking check (LightGBM with and without the parking feature). Superseded by the six-model ablation in Table XIII, produced in `revision_experiments` |
| `Notebook_D_Descriptives.ipynb` | Descriptive statistics: effect of removing duplicates on the cancellation rate, non-refundable duplicates, cancellation rates by attribute, and the change in cancellation rate and guest mix by arrival year (Table XII; Sections III-A, IV-E and V-C) |

Table I summarises published studies and has no code.

## Revision experiments

The `revision_experiments` folder contains the code and outputs for the analyses added in response to the reviewers: fold-fitted encoding, SMOTE-NC and class-weight checks, FT-Transformer and TabNet, and the equal tuning budget (Table VII); the controlled duplicate experiment (Table VIII); the nine two-month expanding windows with bootstrap intervals (Table X, Fig. 4) and booking-date windows; drift diagnostics (Table XI); field-timing checks and the request-field ablation (Section IV-H, Table XIII); and the decision-cost analysis (Table XIV). See `revision_experiments/README.md` for what each script produces and how to run it. All outputs are saved in `revision_experiments/out/`.

## Data

The data are not redistributed here. Download `hotel_bookings.csv` from Kaggle:
https://www.kaggle.com/datasets/qucwang/hotel-bookings-analysis-dataset

Original source: Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

## How to run

Notebooks:

1. Open a notebook in Google Colab (Runtime → Change runtime type → GPU) or in Jupyter with the packages in `requirements.txt`.
2. Put `hotel_bookings.csv` in the same folder as the notebook.
3. Run all cells. Notebooks A and B write their tables (CSV) and figures (PNG, 300 dpi) to `results_A/` and `results_B/`.

Notebooks A and B each take a few hours on a Colab GPU because of the hyperparameter search. Notebook C runs on a CPU in a few minutes and Notebook D in a few seconds.

Revision experiments: put `hotel_bookings.csv` in `revision_experiments/` and run `bash run_all.sh`, then `python field_timing.py`. They run on a CPU; the neural networks take several hours.

## Reproducibility note

The random seed is fixed (42). The neural networks can vary slightly between runs because some GPU operations are not deterministic, and the revision experiments ran on a CPU; the paper reports this (the rerun reproduced the original cross-validation ROC-AUC within 0.005). In the original notebooks the neural networks and XGBoost were trained on the GPU and LightGBM and logistic regression on the CPU, so training times are not comparable across models.

## Licence

Code released under the MIT Licence.
