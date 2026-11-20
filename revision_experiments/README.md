# Revision experiments (IEEE Access resubmission)

Scripts for the experiments added in the revised manuscript. They reuse the cleaning steps of Notebooks A and B,
with categorical encoding fitted inside each training split.

| Script | Produces |
| --- | --- |
| `run_cv.py` | 5-fold CV runs: fold-fitted encoding rerun (Table VII), SMOTE-NC and class weights (Table VII), parking and special-request ablations (Table XIII), controlled duplicate experiment D1/D2/D3 (`--raw`, `--raw --group`; Table VIII) |
| `run_time.py` | nine two-month expanding windows by arrival date (Table X, Fig. 4) and booking-date windows (`--mode booking`) |
| `run_extra.py` | equal-budget boosting (`eqbudget`), FT-Transformer (`ftt`), TabNet (`tabnet`) |
| `drift.py` | PSI, adversarial validation, within-period vs past-trained model, group cancellation rates (Table XI), request-timing check (Section IV-H) |
| `field_timing.py` | time-dependence check of every other input field (Section IV-H); writes `out/field_timing.csv` |
| `analyze.py` | bootstrap CIs, Friedman/Kendall's W, rank correlations, cost analysis (Table XIV); writes `out/results.json` |
| `run_all.sh` | runs everything in order, skipping finished runs |

Put `hotel_bookings.csv` in this folder and run `bash run_all.sh`; it runs every experiment, then `drift.py` and
`analyze.py`. Run `python field_timing.py` separately (seconds). Everything runs on a CPU; the neural networks take several hours.
Extra packages: `pytorch-tabnet` (TabNet only).
