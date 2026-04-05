"""
Data loading utilities for the IBL Behavior Explorer.

Two modes:
  1. Local  — loads pre-saved Parquet files from data/  (default, works offline,
              suitable for Streamlit Community Cloud deployment)
  2. Live   — connects to the IBL ONE API and downloads fresh data

The local Parquet files are generated once by running:
    python scripts/download_data.py
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
TRIALS_FILE = DATA_DIR / "all_trials.parquet"
SUBJECTS_FILE = DATA_DIR / "subjects.txt"


# ─────────────────────────────────────────────────────────────────────
# Local loader  (primary)
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading behavioral dataset…")
def load_local_trials() -> pd.DataFrame:
    """
    Load the pre-saved trials table from data/all_trials.parquet.

    Returns a DataFrame indexed by session UUID with columns including:
        contrastLeft, contrastRight, choice, feedbackType,
        probabilityLeft, reactionTime, training_status,
        performance_easy, training_day, lab, subject, session_start_time
    """
    if not TRIALS_FILE.exists():
        raise FileNotFoundError(
            f"Local data file not found: {TRIALS_FILE}\n"
            "Run  python scripts/download_data.py  to download the dataset first."
        )
    df = pd.read_parquet(TRIALS_FILE)

    # Derived columns used repeatedly downstream
    if "signed_contrast" not in df.columns:
        df = _add_signed_contrast(df)
    # Compute reaction time if not already present
    if "reaction_time" not in df.columns:
        df["reaction_time"] = (df["response_times"] - df["stimOn_times"]).clip(lower=0)
    return df


def load_subject_list() -> list:
    """Return the list of subjects available in the local dataset."""
    if SUBJECTS_FILE.exists():
        return [s.strip() for s in SUBJECTS_FILE.read_text().splitlines() if s.strip()]
    if TRIALS_FILE.exists():
        return sorted(load_local_trials()["subject"].unique().tolist())
    return []


# ─────────────────────────────────────────────────────────────────────
# ONE API live loader  (optional, requires ibllib)
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to IBL ONE API…")
def get_one_client():
    """Return a connected ONE client, or None if ibllib is not installed."""
    try:
        import os as _os
        _os.environ.setdefault("ONE_HTTP_DL_THREADS", "1")
        from one.api import ONE
        ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
        return ONE(password="international")
    except ImportError:
        return None
    except Exception:
        return None


def live_load_subject(subject: str, one=None) -> Optional[pd.DataFrame]:
    """
    Download and return the trials table for a single subject via ONE API.
    Returns None if the download fails.
    """
    if one is None:
        one = get_one_client()
    if one is None:
        return None
    try:
        import sys
        from one.remote.aws import s3_download_file, get_s3_public
        s3, bucket = get_s3_public()

        def _load_agg(subj, dataset):
            if sys.version_info >= (3, 10):
                return one.load_aggregate("subjects", subj, dataset)
            from one.alf.path import add_uuid_string
            files = one.list_aggregates("subjects", subj, dataset=dataset)
            row = files.iloc[0]
            src = str(add_uuid_string(row["rel_path"], row.name))
            dst = one.cache_dir / row["rel_path"]
            local = s3_download_file(src, dst, s3=s3, bucket_name=bucket)
            return pd.read_parquet(local)

        trials = _load_agg(subject, "_ibl_subjectTrials.table.pqt")
        sessions = _load_agg(subject, "_ibl_subjectSessions.table.pqt")
        training = _load_agg(subject, "_ibl_subjectTraining.table.pqt")

        if "task_protocol" in trials.columns:
            trials = trials.drop("task_protocol", axis=1)
        def _as_str_index(df, key_col=None):
            df = df.copy()
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
        drop_cols = [c for c in ["date"] if c in sessions_idx.columns]

        trials = (
            trials_idx
            .join(training_idx)
            .sort_values(by=["session_start_time", "intervals_0"])
        )
        trials["training_status"] = trials.training_status.ffill()
        trials = trials.join(sessions_idx.drop(drop_cols, axis=1))
        trials["subject"] = subject
        trials = _add_signed_contrast(trials)
        trials["reaction_time"] = (trials["response_times"] - trials["stimOn_times"]).clip(lower=0)

        # Add performance_easy and training_day
        from brainbox.behavior.training import compute_performance_easy
        for n_sess, sess in enumerate(trials.index.unique()):
            t = trials[trials.index == sess]
            perf = compute_performance_easy(t)
            trials.loc[trials.index == sess, "performance_easy"] = perf
            trials.loc[trials.index == sess, "training_day"] = n_sess

        return trials
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────

def _add_signed_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed_contrast column: positive = left, negative = right."""
    df = df.copy()
    df["signed_contrast"] = np.nan
    left_mask = df["contrastRight"].isna() | (df["contrastRight"] == 0)
    df.loc[left_mask, "signed_contrast"] = df.loc[left_mask, "contrastLeft"].fillna(0)
    right_mask = ~left_mask
    df.loc[right_mask, "signed_contrast"] = -df.loc[right_mask, "contrastRight"]
    return df


def get_subject_df(all_trials: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Slice trials for a single subject."""
    return all_trials[all_trials["subject"] == subject].copy()


def get_session_df(all_trials: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Slice trials for a single session."""
    if "session" in all_trials.columns:
        return all_trials[all_trials["session"] == session_id].copy()
    return all_trials[all_trials.index == session_id].copy()
