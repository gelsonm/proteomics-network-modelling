"""
Download and save a subset of the IBL behavioral dataset locally.

Usage:
    python scripts/download_data.py [--n-subjects 30]

Downloads the trials and training tables for the first N subjects from
the 2021_Q1_IBL_et_al_Behaviour dataset and saves them as:

    data/all_trials.parquet   — combined trials table
    data/subjects.txt         — list of included subject names

This only needs to be run once.  After that, the Streamlit app loads
from the local parquet file (no internet required).
"""

import argparse
import os
import sys
import warnings

warnings.simplefilter("ignore", FutureWarning)

parser = argparse.ArgumentParser(description="Download IBL behavioral data")
parser.add_argument("--n-subjects", type=int, default=30,
                    help="Number of subjects to download (default: 30)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────
# Verify dependencies
# ─────────────────────────────────────────────────────────────────────
try:
    from one.api import ONE
    from one.remote.aws import s3_download_file, get_s3_public
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install ONE-api ibllib pandas pyarrow")
    sys.exit(1)

try:
    from brainbox.behavior.training import compute_performance_easy
except ImportError:
    # Fallback if brainbox is not installed
    def compute_performance_easy(trials):
        easy = trials[trials["signed_contrast"].abs() >= 0.5]
        if len(easy) == 0:
            return float("nan")
        return float((easy["feedbackType"] == 1).mean())

# ─────────────────────────────────────────────────────────────────────
# Connect to ONE
# ─────────────────────────────────────────────────────────────────────
os.environ.setdefault("ONE_HTTP_DL_THREADS", "1")
ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
one = ONE(password="international")
s3, bucket = get_s3_public()

def load_aggregate(subject, dataset):
    if sys.version_info >= (3, 10):
        return one.load_aggregate("subjects", subject, dataset)
    from one.alf.path import add_uuid_string
    files = one.list_aggregates("subjects", subject, dataset=dataset)
    row = files.iloc[0]
    src = str(add_uuid_string(row["rel_path"], row.name))
    dst = one.cache_dir / row["rel_path"]
    local = s3_download_file(src, dst, s3=s3, bucket_name=bucket)
    return pd.read_parquet(local)

# ─────────────────────────────────────────────────────────────────────
# Find subjects
# ─────────────────────────────────────────────────────────────────────
print("Fetching subject list from IBL…")
datasets = one.alyx.rest(
    "datasets", "list",
    tag="2021_Q1_IBL_et_al_Behaviour",
    name="_ibl_subjectTrials.table.pqt",
)
all_subjects = list({
    d["file_records"][0]["relative_path"].split("/")[2]
    for d in datasets
})
subjects = sorted(all_subjects)[: args.n_subjects]
print(f"Downloading data for {len(subjects)} subjects…\n")

# ─────────────────────────────────────────────────────────────────────
# Download and process
# ─────────────────────────────────────────────────────────────────────
all_trials_list = []

for i, subject in enumerate(subjects):
    print(f"  [{i+1}/{len(subjects)}] {subject}", end="  ")
    try:
        trials   = load_aggregate(subject, "_ibl_subjectTrials.table.pqt")
        sessions = load_aggregate(subject, "_ibl_subjectSessions.table.pqt")
        training = load_aggregate(subject, "_ibl_subjectTraining.table.pqt")

        if "task_protocol" in trials.columns:
            trials = trials.drop("task_protocol", axis=1)

        # Ensure all index / join keys are plain Python str (not Arrow-backed)
        # to avoid pandas 2.x / pyarrow dtype mismatch errors during join.
        def _as_str_index(df, key_col=None):
            df = df.copy()
            # Convert every ArrowDtype / StringDtype column to plain object
            for col in df.columns:
                if hasattr(df[col].dtype, "pyarrow_dtype") or str(df[col].dtype) in ("string", "string[pyarrow]"):
                    df[col] = df[col].astype(object)
            if key_col and key_col in df.columns:
                df[key_col] = df[key_col].astype(str)
                df = df.set_index(key_col)
            else:
                df.index = df.index.astype(str)
            return df

        trials_idx   = _as_str_index(trials,   "session")
        training_idx = _as_str_index(training, "session" if "session" in training.columns else None)
        sessions_idx = _as_str_index(sessions)

        trials = (
            trials_idx
            .join(training_idx)
            .sort_values(by=["session_start_time", "intervals_0"])
        )
        trials["training_status"] = trials["training_status"].ffill()
        drop_cols = [c for c in ["date"] if c in sessions_idx.columns]
        trials = trials.join(sessions_idx.drop(drop_cols, axis=1))
        trials["subject"] = subject

        # signed_contrast
        trials["signed_contrast"] = np.nan
        left_mask = trials["contrastRight"].isna() | (trials["contrastRight"] == 0)
        trials.loc[left_mask, "signed_contrast"] = trials.loc[left_mask, "contrastLeft"].fillna(0)
        trials.loc[~left_mask, "signed_contrast"] = -trials.loc[~left_mask, "contrastRight"]

        # performance_easy and training_day
        for n_sess, sess in enumerate(trials.index.unique()):
            t = trials[trials.index == sess]
            perf = compute_performance_easy(t)
            trials.loc[trials.index == sess, "performance_easy"] = perf
            trials.loc[trials.index == sess, "training_day"] = n_sess

        all_trials_list.append(trials)
        print("✓")

        # Save incrementally every 5 subjects so a crash doesn't lose all work
        if (i + 1) % 5 == 0:
            _out = os.path.join(os.path.dirname(__file__), "..", "data")
            os.makedirs(_out, exist_ok=True)
            pd.concat(all_trials_list).to_parquet(os.path.join(_out, "all_trials.parquet"))
            print(f"    (checkpoint: {i+1} subjects saved)")

    except Exception as ex:
        print(f"✗ ({ex})")
        continue

if not all_trials_list:
    print("No data downloaded. Exiting.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)

all_trials = pd.concat(all_trials_list)
out_path = os.path.join(out_dir, "all_trials.parquet")
all_trials.to_parquet(out_path)
print(f"\nSaved {len(all_trials):,} trials to {out_path}")

subjects_path = os.path.join(out_dir, "subjects.txt")
with open(subjects_path, "w") as f:
    f.write("\n".join(sorted(all_trials["subject"].unique().tolist())))
print(f"Saved subject list to {subjects_path}")
print("\nDone. You can now run:  streamlit run app.py")
