# Duplicate Leakage, Temporal Drift and Model Choice in Hotel Booking Cancellation Prediction

Code for the paper *Duplicate Leakage, Temporal Drift and Model Choice in Hotel Booking Cancellation Prediction: A Leakage-Controlled Comparison of Deep Learning and Gradient Boosting*.

All notebooks are saved with their outputs, so the results can be read without running anything.

| Notebook | What it reproduces |
| --- | --- |
| `Notebook_A_CV_Tuning_SHAP.ipynb` | Duplicate removal, leakage-controlled preprocessing, stratified 5-fold cross-validation of logistic regression, 1D CNN, DNN, CNN+DNN hybrid, XGBoost and LightGBM with hyperparameter search inside each training fold, Friedman and McNemar tests, the duplicate-leakage experiment, and the SHAP analysis (Tables 2, 3, 5–7; Figures 1–3, 5–7) |
| `Notebook_B_Time_Ordered.ipynb` | Expanding-window evaluation on three future periods, with hyperparameter search inside each window (Tables 4 and 8; Figure 4) |
| `Notebook_C_Parking_Check.ipynb` | Sensitivity check: LightGBM with and without the parking feature (Section 4.6) |
| `Notebook_D_Descriptives.ipynb` | Descriptive statistics: effect of removing duplicates on the cancellation rate, non-refundable duplicates, cancellation rates by attribute, and the change in cancellation rate and guest mix by arrival year (Table 9; Sections 3.1, 4.4 and 5.3) |

Table 1 summarises published studies and has no code.

## Data

The data are not redistributed here. Download `hotel_bookings.csv` from Kaggle:
https://www.kaggle.com/datasets/qucwang/hotel-bookings-analysis-dataset

Original source: Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

## How to run

1. Open a notebook in Google Colab (Runtime → Change runtime type → GPU) or in Jupyter with the packages in `requirements.txt`.
2. Put `hotel_bookings.csv` in the same folder as the notebook.
3. Run all cells. Notebooks A and B write their tables (CSV) and figures (PNG, 300 dpi) to `results_A/` and `results_B/`.

Notebooks A and B each take a few hours on a Colab GPU because of the hyperparameter search. Notebook C runs on a CPU in a few minutes and Notebook D in a few seconds.

## Reproducibility note

The random seed is fixed (42). The neural networks can vary slightly between GPU runs because some GPU operations are not deterministic; the paper reports this. The neural networks and XGBoost were trained on the GPU and LightGBM and logistic regression on the CPU, so training times are not comparable across models.

## Licence

Code released under the MIT Licence.
