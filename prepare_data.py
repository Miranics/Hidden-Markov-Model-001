#!/usr/bin/env python3
"""
prepare_data.py
---------------
Turn raw Sensor Logger zip exports into the merged, correctly-named CSV files
that the HMM notebook expects.

WHAT IT DOES
  * scans a folder of Sensor Logger .zip files
  * for each zip: reads Accelerometer.csv + Gyroscope.csv, merges them on their
    shared timestamp into one file with columns:
        timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  * infers the activity from the zip filename (still/standing/walking/jumping)
  * writes numbered files (still_00.csv, still_01.csv, ...) into an output folder

USAGE
  Put your TRAINING zips in one folder and your UNSEEN/test zips in another,
  then run twice:

      python prepare_data.py --in path/to/training_zips  --out data
      python prepare_data.py --in path/to/unseen_zips     --out data/unseen

  (Or edit the paths in the CONFIG block below and just run `python prepare_data.py`.)

NOTES
  * Sensor Logger's Accelerometer.csv columns are: time, seconds_elapsed, z, y, x
    (axes are z,y,x order -- handled below).
  * We use `seconds_elapsed` as the timestamp (seconds from recording start).
  * acc and gyro are recorded on the same clock; if row counts ever differ we
    align by nearest timestamp.
"""
import argparse
import glob
import os
import zipfile
import numpy as np
import pandas as pd

ACTIVITIES = ["still", "standing", "walking", "jumping"]


# ------------------------- CONFIG (used if no CLI args) -------------------------
DEFAULT_IN = "raw_zips"      # folder containing your .zip files
DEFAULT_OUT = "data"         # where merged CSVs are written
# -------------------------------------------------------------------------------


def activity_from_name(fname):
    base = os.path.basename(fname).lower()
    for a in ACTIVITIES:
        if base.startswith(a):
            return a
    return None


def read_sensor(zf, name):
    """Read one sensor CSV from the zip; return DataFrame with t, x, y, z."""
    with zf.open(name) as f:
        df = pd.read_csv(f)
    # Sensor Logger columns: time, seconds_elapsed, z, y, x
    out = pd.DataFrame({
        "t": df["seconds_elapsed"].values,
        "x": df["x"].values,
        "y": df["y"].values,
        "z": df["z"].values,
    })
    return out


def merge_recording(zip_path):
    """Return a merged DataFrame for one recording zip, or None on failure."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if "Accelerometer.csv" not in names or "Gyroscope.csv" not in names:
            print(f"  ! skipping {os.path.basename(zip_path)}: missing Acc/Gyro")
            return None
        acc = read_sensor(zf, "Accelerometer.csv")
        gyro = read_sensor(zf, "Gyroscope.csv")

    # Fast path: identical timestamps and equal length -> direct concat
    if len(acc) == len(gyro) and np.allclose(acc["t"].values, gyro["t"].values,
                                             rtol=0, atol=1e-6):
        merged = pd.DataFrame({
            "timestamp": acc["t"].values,
            "acc_x": acc["x"].values, "acc_y": acc["y"].values, "acc_z": acc["z"].values,
            "gyro_x": gyro["x"].values, "gyro_y": gyro["y"].values, "gyro_z": gyro["z"].values,
        })
        return merged

    # Fallback: align gyro onto acc timestamps by interpolation
    t = acc["t"].values
    merged = pd.DataFrame({"timestamp": t,
                           "acc_x": acc["x"].values,
                           "acc_y": acc["y"].values,
                           "acc_z": acc["z"].values})
    for ax in ["x", "y", "z"]:
        merged[f"gyro_{ax}"] = np.interp(t, gyro["t"].values, gyro[ax].values)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=DEFAULT_IN,
                    help="folder containing Sensor Logger .zip files")
    ap.add_argument("--out", dest="outdir", default=DEFAULT_OUT,
                    help="output folder for merged CSVs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(args.indir, "*.zip")))
    if not zips:
        print(f"No .zip files found in '{args.indir}'.")
        return

    counters = {a: 0 for a in ACTIVITIES}
    written = 0
    unknown = []
    for zp in zips:
        act = activity_from_name(zp)
        if act is None:
            unknown.append(os.path.basename(zp))
            continue
        merged = merge_recording(zp)
        if merged is None:
            continue
        idx = counters[act]
        outname = f"{act}_{idx:02d}.csv"
        merged.to_csv(os.path.join(args.outdir, outname), index=False)
        counters[act] += 1
        written += 1
        print(f"  {os.path.basename(zp):45s} -> {outname}  ({len(merged)} rows)")

    print(f"\nWrote {written} merged CSVs to '{args.outdir}/'")
    for a in ACTIVITIES:
        print(f"  {a:9s}: {counters[a]}")
    if unknown:
        print("\n! Could not infer activity for these (rename so they START "
              "with still/standing/walking/jumping):")
        for u in unknown:
            print("   ", u)


if __name__ == "__main__":
    main()
