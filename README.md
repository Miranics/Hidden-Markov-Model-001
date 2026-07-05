# Modeling Human Activity States Using Hidden Markov Models

Human Activity Recognition (HAR) from a smartphone's accelerometer & gyroscope,
using a Gaussian Hidden Markov Model (Baum-Welch training + Viterbi decoding).

Recorded on an iPhone X with the **Sensor Logger** app at **50 Hz**, four
activities: `still`, `standing`, `walking`, `jumping`.

---

## Quick start (copy-paste)

From inside the project folder:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register the venv as a Jupyter kernel
python -m ipykernel install --user --name hmm-venv --display-name "Python (hmm-venv)"
```

### Add your data

Your Sensor Logger exports are **one zip per recording**. Sort them into the two
`raw_zips` subfolders (just drag them in -- 5 of each activity into `training`,
the rest into `unseen`):

```
raw_zips/
├── training/    <- ~5 zips per activity  (still, standing, walking, jumping)
└── unseen/      <- the remaining test zips (2-3 per activity)
```

Filenames only need to START with the activity word (yours already do, e.g.
`walking_00-2026-07-05_....zip`). The numbers in the filenames don't matter --
the script renumbers each folder cleanly from `00`.

### Build the datasets

```bash
python prepare_data.py --in raw_zips/training --out data
python prepare_data.py --in raw_zips/unseen   --out data/unseen
```

This merges each recording's `Accelerometer.csv` + `Gyroscope.csv` into a single
6-column file (`timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`) and
writes `still_00.csv`, `still_01.csv`, ... into `data/` and `data/unseen/`.

### Run the notebook

```bash
jupyter notebook HMM_Activity_Recognition.ipynb
```

Pick the **Python (hmm-venv)** kernel, confirm the config cell says
`SAMPLING_RATE = 50`, then **Run All**. Figures and the evaluation table
regenerate from your data.

---

## Hardware

Runs fine locally on a CPU-only machine with 8 GB RAM -- no GPU needed. Training
the HMM takes seconds; the whole dataset is only a few MB.

## Repo structure

```
.
├── HMM_Activity_Recognition.ipynb   # main notebook
├── prepare_data.py                  # unzip + merge + rename Sensor Logger exports
├── hmm_pipeline.py                  # reusable feature/HMM functions
├── requirements.txt
├── HMM_Report.docx / .pdf           # 4-5 page report
├── raw_zips/
│   ├── training/                    # <- your training zips go here
│   └── unseen/                      # <- your test zips go here
├── data/                            # generated training CSVs (+ unseen/ subfolder)
└── figures/                         # generated plots + metrics
```

## What's implemented

- Time- & frequency-domain feature extraction over sliding windows
- Gaussian HMM trained with **Baum-Welch** (`hmmlearn`)
- **Viterbi** decoding (both `hmmlearn` and a from-scratch NumPy version)
- Transition-matrix heatmap, decoded-sequence plot, confusion matrix
- Unseen-data evaluation: sensitivity, specificity, per-class accuracy

## After running

The unseen evaluation prints a table and saves it to
`figures/unseen_metrics.csv`. Copy those numbers into the report's results
section and note (from the confusion matrix) which activities got confused.



T
