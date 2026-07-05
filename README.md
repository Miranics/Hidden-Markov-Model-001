# Modeling Human Activity States Using Hidden Markov Models

Human Activity Recognition (HAR) from a smartphone's accelerometer & gyroscope,
using a Gaussian Hidden Markov Model (Baum-Welch training + Viterbi decoding).

Recorded on an **iPhone** with the **Sensor Logger** app at **50 Hz**, for four
activities: `still`, `standing`, `walking`, `jumping`.

## Results summary

The model recognises high-motion activities well and reveals a realistic,
explainable confusion between the two low-motion activities:

| Activity | Sensitivity | Specificity | Accuracy |
|----------|-------------|-------------|----------|
| Still    | 1.00 | 0.89 | 0.92 |
| Standing | 0.00 | 1.00 | 0.75 |
| Walking  | 1.00 | 0.77 | 0.83 |
| Jumping  | 1.00 | 1.00 | 1.00 |

Overall unseen accuracy ~0.75. Jumping, still and walking are recognised
reliably; standing is systematically absorbed into still because the two
low-motion signals are nearly identical once summarised into features. This is
discussed in the report and the notebook's analysis section.

---

## Repo structure

```
.
├── HMM_Activity_Recognition.ipynb   # main notebook (with outputs)
├── prepare_data.py                  # unzip + merge + rename Sensor Logger exports
├── hmm_pipeline.py                  # reusable feature/HMM functions
├── requirements.txt
├── HMM_Report.docx / .pdf           # 4-5 page report
├── raw_zips/
│   ├── training/                    # training zips
│   └── unseen/                      # test zips
├── data/                            # merged training CSVs (+ unseen/ subfolder)
└── figures/                         # exported plots + metrics
```

## Reproduce from scratch

```bash
# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Jupyter kernel (optional; the venv's default python3 kernel also works)
python -m ipykernel install --user --name hmm-venv --display-name "Python (hmm-venv)"

# 4. Build datasets from the raw Sensor Logger zips
python prepare_data.py --in raw_zips/training --out data
python prepare_data.py --in raw_zips/unseen   --out data/unseen

# 5. Run the notebook
jupyter notebook HMM_Activity_Recognition.ipynb   # then Run All
```

The Sensor Logger export gives one zip per recording, each containing separate
`Accelerometer.csv` and `Gyroscope.csv` (columns `time, seconds_elapsed, z, y, x`,
50 Hz). `prepare_data.py` merges them into the 6-column format the notebook needs
(`timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`) and renumbers each
folder from `00`. Filenames only need to START with the activity word.

## Hardware

Runs locally on a CPU-only machine with 16 GB RAM — no GPU needed. Training takes
seconds; the dataset is a few MB.

## Method

- Time- & frequency-domain feature extraction over 1 s sliding windows (50% overlap)
- Gaussian HMM trained with **Baum-Welch** (`hmmlearn`)
- **Viterbi** decoding (library + from-scratch NumPy version, verified to match)
- Transition-matrix heatmap, decoded-sequence plot, confusion matrix
- Unseen-data evaluation: sensitivity, specificity, per-class accuracy


