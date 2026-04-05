"""
Core analyses for the IBL Behavior Explorer.

All functions are pure (no Streamlit calls) so they can be used in both
the app and the companion notebook.

Analyses
--------
1. fit_psychometric        — sigmoid fit to choice ~ contrast
2. compute_block_bias      — P(correct) conditioned on block prior
3. cluster_learning_curves — K-Means on training trajectories
4. choice_decoder          — Logistic regression: predict choice from trial features
5. compute_rt_by_coherence — reaction time as a function of stimulus strength
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────
# 1.  Psychometric curve — sigmoid fit
# ─────────────────────────────────────────────────────────────────────

def _sigmoid(x, alpha, beta, gamma, delta):
    """
    4-parameter psychometric sigmoid.

    P(right) = gamma + (1 - gamma - delta) / (1 + exp(-beta*(x - alpha)))

    alpha : shift (bias, threshold)
    beta  : slope (sensitivity)
    gamma : lower asymptote (lapse low)
    delta : upper asymptote offset  (lapse high)
    """
    return gamma + (1 - gamma - delta) / (1 + np.exp(-beta * (x - alpha)))


def fit_psychometric(
    trials: pd.DataFrame,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fit a 4-parameter sigmoid to P(rightward choice) as a function of
    signed contrast.

    Parameters
    ----------
    trials    : trials DataFrame with columns signed_contrast, choice
    group_col : if provided, fit separately for each value of this column
                (e.g. 'probabilityLeft', 'training_status')

    Returns
    -------
    DataFrame with columns:
        group, alpha (bias), beta (slope), gamma (lapse_low),
        delta (lapse_high), threshold_contrast, raw_x, raw_y, fit_x, fit_y
    """
    def _fit_one(df):
        x = df["signed_contrast"].values
        # choice: 1 = left, 2 = right  →  binary: 1 = rightward
        y = (df["choice"] == 2).astype(float).values

        # Bin means for plotting
        bins = sorted(df["signed_contrast"].unique())
        raw_x, raw_y = [], []
        for b in bins:
            mask = df["signed_contrast"] == b
            raw_x.append(b)
            raw_y.append((df.loc[mask, "choice"] == 2).mean())

        try:
            p0 = [0.0, 5.0, 0.05, 0.05]
            bounds = ([-1.0, 0.1, 0.0, 0.0], [1.0, 50.0, 0.3, 0.3])
            popt, _ = curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds, maxfev=5000)
            alpha, beta, gamma, delta = popt
        except Exception:
            alpha, beta, gamma, delta = 0.0, 5.0, 0.05, 0.05

        fit_x = np.linspace(min(x) - 0.05, max(x) + 0.05, 200)
        fit_y = _sigmoid(fit_x, alpha, beta, gamma, delta)

        # Threshold = contrast at which P(right) = 0.75
        try:
            from scipy.optimize import brentq
            threshold = brentq(lambda c: _sigmoid(c, alpha, beta, gamma, delta) - 0.75,
                               -1.0, 1.0)
        except Exception:
            threshold = np.nan

        return dict(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            threshold_contrast=threshold,
            raw_x=raw_x, raw_y=raw_y,
            fit_x=fit_x.tolist(), fit_y=fit_y.tolist(),
        )

    if group_col is None:
        result = _fit_one(trials)
        result["group"] = "all"
        return pd.DataFrame([result])

    rows = []
    for grp, sub in trials.groupby(group_col):
        if len(sub) < 20:
            continue
        row = _fit_one(sub)
        row["group"] = grp
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# 2.  Block bias — how prior expectation distorts decisions
# ─────────────────────────────────────────────────────────────────────

def compute_block_bias(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P(choose right) conditioned on (signed_contrast, block_prior).

    The IBL biasedChoiceWorld task has blocks where P(stimulus on left) ∈ {0.2, 0.5, 0.8}.
    This analysis asks: does the animal's choice probability shift with the prior,
    even *controlling for* stimulus contrast?

    Returns
    -------
    DataFrame with columns: signed_contrast, probabilityLeft, P_right, n_trials
    """
    df = trials.dropna(subset=["signed_contrast", "probabilityLeft", "choice"]).copy()
    df["chose_right"] = (df["choice"] == 2).astype(float)

    result = (
        df.groupby(["signed_contrast", "probabilityLeft"])
        .agg(P_right=("chose_right", "mean"), n_trials=("chose_right", "count"))
        .reset_index()
    )
    return result


def compute_prior_bias_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single bias metric per subject per block:
    bias = P(choose right | block_right) - P(choose right | block_left)

    Returns DataFrame indexed by subject with: bias, n_right_block, n_left_block.
    """
    df = trials.dropna(subset=["probabilityLeft", "choice", "subject"]).copy()
    df = df[df["probabilityLeft"].isin([0.2, 0.8])]
    df["chose_right"] = (df["choice"] == 2).astype(float)

    rows = []
    for subj, sub in df.groupby("subject"):
        right_block = sub[sub["probabilityLeft"] == 0.2]["chose_right"]
        left_block  = sub[sub["probabilityLeft"] == 0.8]["chose_right"]
        if len(right_block) == 0 or len(left_block) == 0:
            continue
        rows.append(dict(
            subject=subj,
            bias=right_block.mean() - left_block.mean(),
            n_right_block=len(right_block),
            n_left_block=len(left_block),
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# 3.  Learning trajectory clustering
# ─────────────────────────────────────────────────────────────────────

def cluster_learning_curves(
    all_trials: pd.DataFrame,
    n_clusters: int = 3,
    max_days: int = 60,
) -> Tuple[pd.DataFrame, KMeans, np.ndarray]:
    """
    Cluster subjects by their performance trajectory over training days.

    Each subject is represented as a fixed-length vector of daily
    performance-on-easy-trials, interpolated/padded to `max_days` bins.

    Parameters
    ----------
    all_trials : full multi-subject DataFrame
    n_clusters : number of K-Means clusters
    max_days   : trajectory length (days beyond this are ignored)

    Returns
    -------
    subject_df  : DataFrame with columns subject, cluster, trajectory (list)
    km          : fitted KMeans model
    pca_coords  : (n_subjects, 2) array for scatter visualisation
    """
    per_session = (
        all_trials
        .drop_duplicates(subset=["subject", "training_day"])
        [["subject", "training_day", "performance_easy"]]
        .dropna()
    )

    subjects = per_session["subject"].unique()
    X = []
    valid_subjects = []

    for subj in subjects:
        sub = per_session[per_session["subject"] == subj].sort_values("training_day")
        days = sub["training_day"].values.astype(int)
        perf = sub["performance_easy"].values

        # Build fixed-length vector via nearest-day assignment
        traj = np.full(max_days, np.nan)
        for d, p in zip(days, perf):
            if d < max_days:
                traj[d] = p

        # Forward-fill then fill remaining with subject mean
        for i in range(1, max_days):
            if np.isnan(traj[i]):
                traj[i] = traj[i - 1]
        traj = np.where(np.isnan(traj), np.nanmean(traj) if not np.all(np.isnan(traj)) else 0.5, traj)

        X.append(traj)
        valid_subjects.append(subj)

    X = np.array(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    pca2 = PCA(n_components=2)
    pca_coords = pca2.fit_transform(X)

    subject_df = pd.DataFrame({
        "subject":    valid_subjects,
        "cluster":    labels,
        "trajectory": list(X),
    })
    return subject_df, km, pca_coords


# ─────────────────────────────────────────────────────────────────────
# 4.  Choice decoder — logistic regression from trial features
# ─────────────────────────────────────────────────────────────────────

DECODER_FEATURES = [
    "signed_contrast",
    "probabilityLeft",
    "prev_choice",
    "prev_correct",
]


def build_decoder_features(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the feature matrix for the choice decoder.

    Features
    --------
    signed_contrast  : stimulus evidence (-1 = right, +1 = left)
    probabilityLeft  : block prior (0.2 / 0.5 / 0.8)
    prev_choice      : choice on the previous trial (-1 = right, 1 = left, 0 = first)
    prev_correct     : feedback on previous trial (1 = correct, -1 = incorrect, 0 = first)
    """
    df = trials.dropna(subset=["signed_contrast", "probabilityLeft", "choice"]).copy()
    # Map choice to binary (1 = left / 2 = right → 0/1)
    df["choice_bin"] = (df["choice"] == 2).astype(int)

    df["prev_choice"] = df["choice_bin"].shift(1).fillna(0.5)
    df["prev_correct"] = (
        df["feedbackType"].shift(1).fillna(0).map({1: 1, -1: -1, 0: 0})
    )

    # Drop trials where any feature is NaN
    df = df.dropna(subset=DECODER_FEATURES)
    return df


def fit_choice_decoder(
    trials: pd.DataFrame,
    test_frac: float = 0.25,
    random_state: int = 42,
) -> Dict:
    """
    Fit a logistic regression decoder to predict the animal's choice.

    Returns a dict with:
        model         : fitted LogisticRegression
        scaler        : fitted StandardScaler
        accuracy      : float
        auc           : float
        feature_importance : dict {feature: coefficient}
        y_true, y_pred_prob : arrays for ROC curve
    """
    df = build_decoder_features(trials)
    if len(df) < 50:
        return {}

    X = df[DECODER_FEATURES].values
    y = df["choice_bin"].values

    # Train / test split (preserve temporal order)
    n_test = max(1, int(len(X) * test_frac))
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    y_prob   = model.predict_proba(X_test)[:, 1]
    accuracy = float(np.mean(y_pred == y_test))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float("nan")

    coef = model.coef_[0]
    feature_importance = dict(zip(DECODER_FEATURES, coef.tolist()))

    return dict(
        model=model,
        scaler=scaler,
        accuracy=accuracy,
        auc=auc,
        feature_importance=feature_importance,
        y_true=y_test.tolist(),
        y_pred_prob=y_prob.tolist(),
        n_train=len(X_train),
        n_test=len(X_test),
    )


# ─────────────────────────────────────────────────────────────────────
# 5.  Reaction time by coherence
# ─────────────────────────────────────────────────────────────────────

def compute_rt_by_contrast(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± SEM reaction time for each signed contrast level.

    Returns DataFrame with: signed_contrast, mean_rt, sem_rt, n.
    """
    df = trials.dropna(subset=["signed_contrast", "reaction_time"]).copy()
    df = df[df["reaction_time"] > 0]   # exclude invalid RTs

    result = (
        df.groupby("signed_contrast")["reaction_time"]
        .agg(mean_rt="mean", sem_rt="sem", n="count")
        .reset_index()
    )
    return result
