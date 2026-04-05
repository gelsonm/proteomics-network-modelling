"""
Plotly-based visualisation utilities for the IBL Behavior Explorer.
All functions return go.Figure objects for use in Streamlit.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Colour palette (consistent across all figures)
_PALETTE = {
    "left_block":  "#e63946",   # red  — P(left) = 0.8
    "neutral":     "#457b9d",   # blue — P(left) = 0.5
    "right_block": "#2a9d8f",   # teal — P(left) = 0.2
    "correct":     "#2a9d8f",
    "error":       "#e63946",
    "primary":     "#1b2a4a",
    "light":       "#a8b5c8",
}

_BLOCK_COLOR = {0.8: _PALETTE["left_block"],
                0.5: _PALETTE["neutral"],
                0.2: _PALETTE["right_block"]}
_BLOCK_LABEL = {0.8: "P(left) = 0.8  (left block)",
                0.5: "P(left) = 0.5  (unbiased)",
                0.2: "P(left) = 0.2  (right block)"}


# ─────────────────────────────────────────────────────────────────────
# 1.  Psychometric curve
# ─────────────────────────────────────────────────────────────────────

def plot_psychometric(
    psych_df: pd.DataFrame,
    title: str = "Psychometric Function",
) -> go.Figure:
    """
    Scatter of binned proportions + fitted sigmoid line(s).

    psych_df : output of analysis.fit_psychometric()
    """
    fig = go.Figure()

    for _, row in psych_df.iterrows():
        grp  = row["group"]
        col  = _BLOCK_COLOR.get(grp, _PALETTE["primary"])
        lbl  = _BLOCK_LABEL.get(grp, str(grp))

        # Sigmoid fit
        fig.add_trace(go.Scatter(
            x=row["fit_x"], y=row["fit_y"],
            mode="lines",
            line=dict(color=col, width=2.5),
            name=lbl,
            legendgroup=str(grp),
            showlegend=True,
        ))
        # Binned data points
        fig.add_trace(go.Scatter(
            x=row["raw_x"], y=row["raw_y"],
            mode="markers",
            marker=dict(color=col, size=9, symbol="circle"),
            legendgroup=str(grp),
            showlegend=False,
            hovertemplate="Contrast=%{x:.3f}<br>P(right)=%{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="Chance (0.5)")
    fig.update_layout(
        title=title,
        xaxis_title="Signed Contrast  (negative = right side)",
        yaxis_title="P(chose right)",
        yaxis=dict(range=[-0.05, 1.05]),
        height=420,
        margin=dict(t=60),
        legend=dict(title="Block / Condition"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 2.  Block bias visualisation
# ─────────────────────────────────────────────────────────────────────

def plot_block_bias(block_df: pd.DataFrame) -> go.Figure:
    """
    P(chose right) vs signed contrast, one line per block prior.

    block_df : output of analysis.compute_block_bias()
    """
    fig = go.Figure()

    for prob_left in sorted(block_df["probabilityLeft"].unique()):
        sub = block_df[block_df["probabilityLeft"] == prob_left].sort_values("signed_contrast")
        col = _BLOCK_COLOR.get(prob_left, _PALETTE["light"])
        lbl = _BLOCK_LABEL.get(prob_left, f"P(L)={prob_left}")

        fig.add_trace(go.Scatter(
            x=sub["signed_contrast"].tolist(),
            y=sub["P_right"].tolist(),
            mode="lines+markers",
            line=dict(color=col, width=2.5),
            marker=dict(
                color=col,
                size=sub["n_trials"].clip(5, 200).tolist(),
                sizemode="area", sizeref=2,
            ),
            name=lbl,
            hovertemplate=(
                "Contrast=%{x:.3f}<br>P(right)=%{y:.3f}<br>"
                "n=%{customdata}<extra></extra>"
            ),
            customdata=sub["n_trials"].tolist(),
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="Unbiased")
    fig.add_vline(x=0.0, line_dash="dash", line_color="#cccccc")
    fig.update_layout(
        title="Block-Prior Bias: How Internal State Shifts Decisions",
        xaxis_title="Signed Contrast  (positive = stimulus on left)",
        yaxis_title="P(chose right)",
        yaxis=dict(range=[-0.02, 1.02]),
        height=440,
        margin=dict(t=65),
        legend=dict(title="Block prior"),
    )
    return fig


def plot_prior_bias_distribution(bias_df: pd.DataFrame) -> go.Figure:
    """
    Histogram of per-subject bias scores (P_right_block - P_left_block).
    A positive bias → follows the block prior; negative → doesn't.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bias_df["bias"].tolist(),
        nbinsx=20,
        marker_color=_PALETTE["primary"],
        opacity=0.8,
        name="Bias score",
    ))
    fig.add_vline(x=0.0, line_dash="dash", line_color=_PALETTE["error"],
                  annotation_text="No bias")
    fig.update_layout(
        title="Distribution of Prior Bias Across Subjects",
        xaxis_title="Bias  =  P(right|right block) − P(right|left block)",
        yaxis_title="Number of subjects",
        height=360,
        margin=dict(t=55),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 3.  Learning curves
# ─────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    all_trials: pd.DataFrame,
    highlight_subject: Optional[str] = None,
    max_days: int = 90,
) -> go.Figure:
    """
    Individual learning trajectories (performance on easy trials vs day).
    One line per subject; optionally highlight one subject.
    """
    per_session = (
        all_trials
        .drop_duplicates(subset=["subject", "training_day"])
        [["subject", "training_day", "performance_easy", "lab"]]
        .dropna()
    )

    fig = go.Figure()
    for subj, sub in per_session.groupby("subject"):
        sub = sub.sort_values("training_day")
        sub = sub[sub["training_day"] <= max_days]
        is_hl = (subj == highlight_subject)
        fig.add_trace(go.Scatter(
            x=sub["training_day"].tolist(),
            y=(sub["performance_easy"] * 100).tolist(),
            mode="lines",
            line=dict(
                color=_PALETTE["primary"] if is_hl else _PALETTE["light"],
                width=3 if is_hl else 1,
            ),
            opacity=1.0 if is_hl else 0.35,
            name=subj if is_hl else None,
            showlegend=is_hl,
            hovertemplate=f"{subj}<br>Day=%{{x}}<br>Perf=%{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=80, line_dash="dot", line_color=_PALETTE["correct"],
                  annotation_text="80 % criterion")
    fig.update_layout(
        title="Learning Trajectories Across All Subjects",
        xaxis_title="Training Day",
        yaxis_title="Performance on Easy Trials (%)",
        yaxis=dict(range=[30, 105]),
        height=420,
        margin=dict(t=60),
        showlegend=bool(highlight_subject),
    )
    return fig


def plot_clustered_trajectories(
    subject_df: pd.DataFrame,
    pca_coords: np.ndarray,
    n_clusters: int,
) -> go.Figure:
    """
    Two-panel figure:
    Left  — mean learning trajectory per cluster
    Right — PCA scatter of trajectory vectors, coloured by cluster
    """
    cluster_colors = px.colors.qualitative.Set2[:n_clusters]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Mean Learning Curve per Cluster",
            "Trajectory Clustering  (PCA of 60-day vectors)",
        ],
        horizontal_spacing=0.12,
    )

    # ── Mean trajectories ─────────────────────────────────────────
    for c in range(n_clusters):
        mask = subject_df["cluster"] == c
        trajs = np.stack(subject_df.loc[mask, "trajectory"].values)
        mean_traj = trajs.mean(axis=0)
        sem_traj  = trajs.std(axis=0) / np.sqrt(len(trajs))
        days = list(range(len(mean_traj)))

        fig.add_trace(go.Scatter(
            x=days, y=(mean_traj * 100).tolist(),
            mode="lines",
            line=dict(color=cluster_colors[c], width=2.5),
            name=f"Cluster {c+1}  (n={mask.sum()})",
            legendgroup=str(c),
        ), row=1, col=1)

        # SEM band
        fig.add_trace(go.Scatter(
            x=days + days[::-1],
            y=((mean_traj + sem_traj) * 100).tolist()
             + ((mean_traj - sem_traj) * 100).tolist()[::-1],
            fill="toself",
            fillcolor=cluster_colors[c].replace(")", ",0.15)").replace("rgb", "rgba"),
            line=dict(width=0),
            showlegend=False,
            legendgroup=str(c),
        ), row=1, col=1)

    # ── PCA scatter ───────────────────────────────────────────────
    for c in range(n_clusters):
        mask = subject_df["cluster"] == c
        pts  = pca_coords[mask]
        fig.add_trace(go.Scatter(
            x=pts[:, 0].tolist(), y=pts[:, 1].tolist(),
            mode="markers",
            marker=dict(color=cluster_colors[c], size=9, opacity=0.8),
            name=f"Cluster {c+1}",
            legendgroup=str(c),
            showlegend=False,
            hovertext=subject_df.loc[mask, "subject"].tolist(),
            hovertemplate="%{hovertext}<extra></extra>",
        ), row=1, col=2)

    fig.update_xaxes(title_text="Training Day",      row=1, col=1)
    fig.update_yaxes(title_text="Performance (%)",   row=1, col=1, range=[30, 105])
    fig.update_xaxes(title_text="PC 1",              row=1, col=2)
    fig.update_yaxes(title_text="PC 2",              row=1, col=2)
    fig.update_layout(height=430, margin=dict(t=65))
    return fig


# ─────────────────────────────────────────────────────────────────────
# 4.  Choice decoder
# ─────────────────────────────────────────────────────────────────────

def plot_decoder_results(result: Dict) -> go.Figure:
    """
    Two-panel: feature coefficients + ROC curve.
    """
    from sklearn.metrics import roc_curve

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Logistic Regression Coefficients",
            f"ROC Curve  (AUC = {result['auc']:.3f})",
        ],
        horizontal_spacing=0.14,
    )

    # ── Coefficients ──────────────────────────────────────────────
    fi   = result["feature_importance"]
    feat = list(fi.keys())
    coef = list(fi.values())
    colors = [_PALETTE["correct"] if c > 0 else _PALETTE["error"] for c in coef]

    fig.add_trace(go.Bar(
        x=coef, y=feat,
        orientation="h",
        marker_color=colors,
        name="Coefficient",
    ), row=1, col=1)
    fig.add_vline(x=0, line_color="gray", line_dash="dash", row=1, col=1)

    # ── ROC curve ─────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(result["y_true"], result["y_pred_prob"])
    fig.add_trace(go.Scatter(
        x=fpr.tolist(), y=tpr.tolist(),
        mode="lines",
        line=dict(color=_PALETTE["primary"], width=2.5),
        name="ROC",
        fill="tozeroy",
        fillcolor="rgba(27,42,74,0.10)",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Chance",
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Coefficient value",       row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate",     row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate",      row=1, col=2,
                     range=[-0.02, 1.02])
    fig.update_layout(height=390, margin=dict(t=60), showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 5.  Reaction time by contrast
# ─────────────────────────────────────────────────────────────────────

def plot_rt_by_contrast(rt_df: pd.DataFrame) -> go.Figure:
    """
    Mean ± SEM reaction time vs signed contrast.
    Typically a V-shape: fast at high contrast, slow near zero.
    """
    rt_df = rt_df.sort_values("signed_contrast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rt_df["signed_contrast"].tolist(),
        y=rt_df["mean_rt"].tolist(),
        mode="lines+markers",
        error_y=dict(
            type="data",
            array=rt_df["sem_rt"].tolist(),
            visible=True,
            color=_PALETTE["primary"],
        ),
        line=dict(color=_PALETTE["primary"], width=2.5),
        marker=dict(size=9),
        name="Mean RT",
    ))

    fig.update_layout(
        title="Reaction Time vs Stimulus Contrast",
        xaxis_title="Signed Contrast  (negative = right side)",
        yaxis_title="Reaction Time (s)",
        height=380,
        margin=dict(t=55),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 6.  Cross-subject performance comparison (lab/group level)
# ─────────────────────────────────────────────────────────────────────

def plot_cross_subject_perf(all_trials: pd.DataFrame) -> go.Figure:
    """
    Violin plot of peak performance-on-easy-trials per subject, grouped by lab.
    """
    peak = (
        all_trials
        .groupby("subject")
        .agg(peak_perf=("performance_easy", "max"),
             lab=("lab", "first"))
        .reset_index()
        .dropna()
    )
    peak["peak_perf_pct"] = peak["peak_perf"] * 100

    fig = px.violin(
        peak, x="lab", y="peak_perf_pct",
        box=True, points="all",
        color="lab",
        labels={"peak_perf_pct": "Peak Performance (%)", "lab": "Lab"},
        title="Peak Performance Distribution Across Labs",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_hline(y=80, line_dash="dot", line_color="gray",
                  annotation_text="80 % criterion")
    fig.update_layout(height=420, margin=dict(t=60), showlegend=False)
    return fig
