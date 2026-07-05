"""
hmm_pipeline.py
---------------
Reusable functions for the Human Activity Recognition HMM project.
Imported by the notebook; also runnable standalone for a quick sanity check.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq

ACTIVITIES = ["still", "standing", "walking", "jumping"]
ACT2ID = {a: i for i, a in enumerate(ACTIVITIES)}

WINDOW_SEC = 1.0      # window length in seconds
OVERLAP = 0.5         # 50% overlap


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load one recording, return DataFrame with the 6 sensor axes."""
    df = pd.read_csv(path)
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}. Found: {list(df.columns)}")
    return df


def label_from_filename(path):
    base = os.path.basename(path).lower()
    for a in ACTIVITIES:
        if base.startswith(a):
            return a
    raise ValueError(f"Cannot infer activity from filename: {base}")


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------
def _window_features(win, fs):
    """Compute time- and frequency-domain features for one window.
    win: (n_samples, 6) array [ax, ay, az, gx, gy, gz]."""
    feats = {}
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]

    # ---- Time-domain ----
    for j, name in enumerate(axes):
        x = win[:, j]
        feats[f"{name}_mean"] = np.mean(x)
        feats[f"{name}_std"] = np.std(x)
        feats[f"{name}_var"] = np.var(x)

    # Signal magnitude area (accelerometer) - mean abs summed over axes
    acc = win[:, 0:3]
    feats["acc_sma"] = np.mean(np.sum(np.abs(acc), axis=1))
    # Resultant acc magnitude stats
    acc_mag = np.linalg.norm(acc, axis=1)
    feats["acc_mag_mean"] = np.mean(acc_mag)
    feats["acc_mag_std"] = np.std(acc_mag)

    # Correlation between accelerometer axes
    def corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return np.corrcoef(a, b)[0, 1]
    feats["corr_xy"] = corr(acc[:, 0], acc[:, 1])
    feats["corr_xz"] = corr(acc[:, 0], acc[:, 2])
    feats["corr_yz"] = corr(acc[:, 1], acc[:, 2])

    # ---- Frequency-domain (on acc magnitude) ----
    n = len(acc_mag)
    detr = acc_mag - np.mean(acc_mag)
    fft_vals = np.abs(rfft(detr))
    freqs = rfftfreq(n, d=1.0 / fs)
    if len(fft_vals) > 1:
        dom_idx = np.argmax(fft_vals[1:]) + 1  # skip DC
        feats["dom_freq"] = freqs[dom_idx]
        feats["spectral_energy"] = np.sum(fft_vals ** 2) / n
    else:
        feats["dom_freq"] = 0.0
        feats["spectral_energy"] = 0.0

    return feats


def extract_features_from_df(df, fs, label=None):
    """Slide a window over a recording, return a DataFrame of feature rows."""
    data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values
    win_n = int(WINDOW_SEC * fs)
    step = max(1, int(win_n * (1 - OVERLAP)))
    rows = []
    for start in range(0, len(data) - win_n + 1, step):
        win = data[start:start + win_n]
        f = _window_features(win, fs)
        if label is not None:
            f["label"] = label
            f["label_id"] = ACT2ID[label]
        rows.append(f)
    return pd.DataFrame(rows)


def build_dataset(data_dir, fs):
    """Read every CSV in data_dir, return (features_df, list_of_sequences).
    Each sequence = feature rows from one recording (kept together for HMM)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    all_feats, sequences = [], []
    for path in files:
        label = label_from_filename(path)
        df = load_csv(path)
        fdf = extract_features_from_df(df, fs, label)
        if len(fdf) == 0:
            continue
        all_feats.append(fdf)
        sequences.append(fdf)
    full = pd.concat(all_feats, ignore_index=True)
    return full, sequences


# ---------------------------------------------------------------------------
# 3. Evaluation helpers
# ---------------------------------------------------------------------------
def per_class_metrics(y_true, y_pred, n_states=4):
    """Return sensitivity, specificity, and per-class accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for s in range(n_states):
        tp = np.sum((y_pred == s) & (y_true == s))
        fn = np.sum((y_pred != s) & (y_true == s))
        tn = np.sum((y_pred != s) & (y_true != s))
        fp = np.sum((y_pred == s) & (y_true != s))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        acc = (tp + tn) / len(y_true)
        n = int(np.sum(y_true == s))
        out[s] = dict(n=n, sensitivity=sens, specificity=spec, accuracy=acc)
    return out


def align_states_to_labels(pred_states, true_labels, n_states=4):
    """HMM states are unordered; map each learned state to the true label
    it most overlaps with (Hungarian-style greedy on the confusion counts)."""
    from itertools import permutations
    best_perm, best_acc = None, -1
    for perm in permutations(range(n_states)):
        mapped = np.array([perm[s] for s in pred_states])
        acc = np.mean(mapped == true_labels)
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    mapped = np.array([best_perm[s] for s in pred_states])
    return mapped, best_perm


if __name__ == "__main__":
    from hmmlearn import hmm
    fs = 50
    full, seqs = build_dataset("data", fs)
    print("Feature matrix:", full.shape)
    print("Columns:", [c for c in full.columns if c not in ("label", "label_id")][:8], "...")

    feat_cols = [c for c in full.columns if c not in ("label", "label_id")]
    X = full[feat_cols].values
    lengths = [len(s) for s in seqs]

    # standardize
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd

    model = hmm.GaussianHMM(n_components=4, covariance_type="diag",
                            n_iter=100, random_state=42)
    model.fit(Xn, lengths)
    states = model.predict(Xn, lengths)
    mapped, perm = align_states_to_labels(states, full["label_id"].values)
    acc = np.mean(mapped == full["label_id"].values)
    print(f"Train accuracy after state alignment: {acc:.3f}")
    m = per_class_metrics(full["label_id"].values, mapped)
    for s, d in m.items():
        print(ACTIVITIES[s], d)
