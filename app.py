"""
IBL Behavior Explorer — Streamlit Application
==============================================

Analyses of mouse decision-making behavior from the IBL brain-wide map
dataset, with a focus on how internal state (block prior) shapes choice,
cross-subject learning trajectory clustering, and behavioral decoding.

Run with:
    streamlit run app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import load_local_trials, load_subject_list, get_subject_df
from src.analysis import (
    fit_psychometric,
    compute_block_bias,
    compute_prior_bias_summary,
    cluster_learning_curves,
    fit_choice_decoder,
    compute_rt_by_contrast,
)
from src.visualization import (
    plot_psychometric,
    plot_block_bias,
    plot_prior_bias_distribution,
    plot_learning_curves,
    plot_clustered_trajectories,
    plot_decoder_results,
    plot_rt_by_contrast,
    plot_cross_subject_perf,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBL Behavior Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background-color: #f5f7fa; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
h1 { color: #0d1b2a; font-weight: 800; letter-spacing: -0.5px; }
h2, h3 { color: #1b2a4a; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #e8ecf3; border-radius: 8px 8px 0 0;
    padding: 6px 18px; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #1b2a4a !important; color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## IBL Behavior Explorer")
    st.caption("International Brain Laboratory · Brain-Wide Map Dataset")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Psychometric Analysis",
            "Block Prior & Internal State",
            "Learning Dynamics",
            "Choice Decoder",
            "About",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("####Data")

# ─────────────────────────────────────────────────────────────────────
# Load data (with graceful error handling)
# ─────────────────────────────────────────────────────────────────────
DATA_LOADED = False
all_trials = None
subject_list = []

try:
    all_trials   = load_local_trials()
    subject_list = sorted(all_trials["subject"].unique().tolist())
    DATA_LOADED  = True
    with st.sidebar:
        st.success(f" {len(subject_list)} subjects loaded")
        st.caption(f"{len(all_trials):,} trials total")
except FileNotFoundError:
    with st.sidebar:
        st.warning(
            "No local data found.  \n"
            "Run `python scripts/download_data.py` first, "
            "or switch to Live mode below."
        )

# ─────────────────────────────────────────────────────────────────────
# Subject picker (sidebar, shown on relevant pages)
# ─────────────────────────────────────────────────────────────────────
selected_subject = None
if DATA_LOADED and ("Analysis" in page or "Decoder" in page or "Block" in page or "Internal" in page):
    with st.sidebar:
        selected_subject = st.selectbox(
            "Subject (for single-animal views)",
            subject_list,
            index=0,
        )

# ─────────────────────────────────────────────────────────────────────
# Helper: require data
# ─────────────────────────────────────────────────────────────────────
def require_data():
    if not DATA_LOADED:
        st.warning(
            " No data available.  \n"
            "Download the dataset by running `python scripts/download_data.py` "
            "from the project root."
        )
        st.stop()


# ══════════════════════════════════════════════════════════════════════
# PAGE ▸ Overview
# ══════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("IBL Brain-Wide Behavior Explorer")
    st.markdown(
        "An analysis of decision-making behavior from the "
        "[International Brain Laboratory](https://www.internationalbrainlab.com/) "
        "Brain-Wide Map dataset — 459 mice, 12 labs across 3 continents, "
        "standardised 2AFC perceptual decision task."
    )
    st.divider()

    if DATA_LOADED:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subjects",  len(subject_list))
        c2.metric("Total trials", f"{len(all_trials):,}")
        labs = all_trials["lab"].nunique() if "lab" in all_trials.columns else "—"
        c3.metric("Labs", labs)
        sessions = all_trials.index.nunique() if all_trials.index.name else "—"
        c4.metric("Sessions", sessions if isinstance(sessions, str) else f"{sessions:,}")

    st.divider()

    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown(
            """
### What this app analyses

| Page | Scientific question |
|---|---|
| Psychometric Analysis | How does stimulus strength (contrast) govern choice accuracy? |
| Block Prior & Internal State | Does the animal's *prior expectation* — an internal state — bias its choices *independent of* stimulus evidence? |
| Learning Dynamics | How do subjects learn and can we identify distinct learning strategies by clustering trajectories? |
| Choice Decoder | Can we predict the animal's next choice from behavioral history alone, and which features matter most? |
"""
        )
    with c_right:
        st.markdown(
            """
### Task structure

The IBL 2AFC task presents visual stimuli of variable contrast on either the
left or right monitor.  Within a session, **blocks** alternate between periods
where the stimulus is more likely to appear on the left (P = 0.8) or right
(P = 0.2).  Animals learn this block structure and use it as an internal prior.

```
   ┌──────────┬──────────┬──────────┐
   │ Block L  │ Block R  │ Block L  │  ← hidden from experimenter
   │ P(L)=0.8 │ P(L)=0.2 │ P(L)=0.8 │
   └──────────┴──────────┴──────────┘
        ↓           ↓
   Mice must integrate bottom-up evidence + top-down expectations
```
"""
        )

    st.info(
        "**Relevance to large-scale systems neuroscience:** This task is deployed "
        "across 12 labs with simultaneous Neuropixels recordings (brain-wide map). "
        "The behavioral data here captures the *internal state dynamics* — "
        "the hidden block prior — that the neural population must encode "
        "to guide decisions."
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE ▸ Psychometric Analysis
# ══════════════════════════════════════════════════════════════════════
elif "Psychometric" in page:
    st.title("Psychometric Analysis")
    require_data()

    st.markdown(
        "The psychometric function describes how choice accuracy scales with "
        "stimulus strength (contrast). We fit a **4-parameter sigmoid**:"
    )
    st.latex(
        r"P(\text{right}) = \gamma + "
        r"\frac{1 - \gamma - \delta}{1 + e^{-\beta (c - \alpha)}}"
    )
    st.markdown(
        r"where $\alpha$ = bias (threshold), $\beta$ = slope (sensitivity), "
        r"$\gamma, \delta$ = lapse rates."
    )
    st.divider()

    tab_single, tab_by_block, tab_rt = st.tabs(
        ["Single Subject", "Split by Block Prior", "Reaction Time"]
    )

    # ── Single subject ────────────────────────────────────────────────
    with tab_single:
        with st.sidebar:
            selected_subject = st.selectbox(
                "Subject", subject_list, key="psych_subj"
            ) if "psych_subj" not in st.session_state else subject_list[0]

        subj_ctrl, _ = st.columns([1, 2])
        with subj_ctrl:
            sel = st.selectbox("Select subject", subject_list, key="psych_sel")

        subj_df = get_subject_df(all_trials, sel)
        psych = fit_psychometric(subj_df)
        st.plotly_chart(plot_psychometric(psych, title=f"Psychometric Function — {sel}"),
                        use_container_width=True)

        # Summary metrics
        row = psych.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bias α", f"{row['alpha']:.3f}",
                  help="Contrast at which P(right) = 0.5. Negative = left bias.")
        m2.metric("Slope β", f"{row['beta']:.2f}",
                  help="Sensitivity to contrast. Higher = steeper.")
        m3.metric("Lapse γ+δ", f"{(row['gamma']+row['delta']):.3f}",
                  help="Fraction of trials incorrectly answered even at max contrast.")
        m4.metric("Threshold", f"{row['threshold_contrast']:.3f}",
                  help="Contrast at which P(right) = 0.75.")

    # ── Block prior comparison ────────────────────────────────────────
    with tab_by_block:
        st.markdown(
            "Fit separate psychometric curves for each **block condition** to see "
            "how the internal prior shifts the entire curve horizontally."
        )
        ctrl_col, _ = st.columns([1, 2])
        with ctrl_col:
            sel2 = st.selectbox("Select subject", subject_list, key="psych_blk")
        subj_df2 = get_subject_df(all_trials, sel2)
        # Only biasedChoiceWorld sessions have block structure
        biased = subj_df2.dropna(subset=["probabilityLeft"])
        biased = biased[biased["probabilityLeft"].isin([0.2, 0.8])]
        if len(biased) < 100:
            st.info(f"{sel2} has insufficient biased-block trials (< 100). "
                    "Try a different subject.")
        else:
            psych_blk = fit_psychometric(biased, group_col="probabilityLeft")
            st.plotly_chart(
                plot_psychometric(psych_blk,
                                  title=f"Block-Stratified Psychometric — {sel2}"),
                use_container_width=True,
            )
            with st.expander("What does this show?"):
                st.markdown(
                    "A **horizontal shift** of the curve between blocks (same slope, "
                    "different $\\alpha$) indicates the animal has learned the block "
                    "prior and uses it as an internal prior belief — exactly what "
                    "Bayes-optimal behaviour predicts.  "
                    "The shift in $\\alpha$ quantifies the *strength of prior inference*."
                )

    # ── Reaction time ─────────────────────────────────────────────────
    with tab_rt:
        st.markdown(
            "Reaction time typically shows a **V-shape** vs contrast: "
            "fast at high contrast (easy), slow near zero (ambiguous). "
            "This mirrors the drift-diffusion prediction that evidence accumulation "
            "time depends on SNR."
        )
        ctrl_col_rt, _ = st.columns([1, 2])
        with ctrl_col_rt:
            sel_rt = st.selectbox("Select subject", subject_list, key="rt_sel")
        subj_rt = get_subject_df(all_trials, sel_rt)
        rt_df = compute_rt_by_contrast(subj_rt)
        if rt_df.empty:
            st.info("No reaction time data available for this subject.")
        else:
            st.plotly_chart(plot_rt_by_contrast(rt_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE ▸ Block Prior & Internal State
# ══════════════════════════════════════════════════════════════════════
elif "Block" in page or "Internal" in page:
    st.title("Block Prior & Internal State Encoding")
    require_data()

    st.markdown(
        """
The IBL task embeds a **hidden Markov block structure**: the side where the stimulus
appears is drawn from an alternating prior (P = 0.8 or 0.2).  Animals that learn
this structure develop an **internal state** — a prior belief about the likely
stimulus side — that shifts their choices beyond what the sensory evidence alone
prescribes.

This is directly analogous to the homeostatic internal states (hunger, fear,
social drive) studied in systems neuroscience: a hidden variable that
modulates the animal's behavioural policy.
"""
    )
    st.divider()

    tab_block, tab_dist, tab_pop = st.tabs(
        ["Single Subject", "Bias Distribution", "Cross-Lab Comparison"]
    )

    # ── Single subject block bias ────────────────────────────────────
    with tab_block:
        ctrl, _ = st.columns([1, 2])
        with ctrl:
            sel_blk = st.selectbox("Select subject", subject_list, key="blk_sel")
        subj_blk = get_subject_df(all_trials, sel_blk)
        biased_blk = subj_blk.dropna(subset=["probabilityLeft", "signed_contrast"])
        biased_blk = biased_blk[biased_blk["probabilityLeft"].isin([0.2, 0.5, 0.8])]

        if len(biased_blk) < 50:
            st.info("Insufficient biased-block data for this subject.")
        else:
            block_df = compute_block_bias(biased_blk)
            st.plotly_chart(plot_block_bias(block_df), use_container_width=True)

            # Bias delta at 0% contrast
            zero = block_df[block_df["signed_contrast"].abs() < 0.01]
            if len(zero) >= 2:
                right_block = zero[zero["probabilityLeft"] == 0.2]["P_right"]
                left_block  = zero[zero["probabilityLeft"] == 0.8]["P_right"]
                if len(right_block) and len(left_block):
                    delta = float(right_block.values[0] - left_block.values[0])
                    st.metric(
                        "Prior bias at 0% contrast  (Δ P_right)",
                        f"{delta:+.3f}",
                        help="Difference in P(right) between right-block and left-block "
                             "trials at zero stimulus contrast — pure prior influence.",
                    )

    # ── Population bias distribution ─────────────────────────────────
    with tab_dist:
        st.markdown(
            "For each subject we compute a **bias score** = "
            "P(right | right block) − P(right | left block) at averaged across contrasts.  "
            "Scores > 0 indicate the animal follows the block prior; "
            "near 0 indicates no learning of the block structure."
        )
        with st.spinner("Computing bias across all subjects…"):
            bias_df = compute_prior_bias_summary(all_trials)

        if bias_df.empty:
            st.info("No biased-block data found in the dataset.")
        else:
            st.plotly_chart(plot_prior_bias_distribution(bias_df),
                            use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Subjects with bias data",  len(bias_df))
            m2.metric("Mean bias score",   f"{bias_df['bias'].mean():.3f}")
            m3.metric("Subjects with bias > 0.1",
                      int((bias_df["bias"] > 0.1).sum()))

    # ── Cross-lab ─────────────────────────────────────────────────────
    with tab_pop:
        st.markdown(
            "Comparing peak performance across labs validates the "
            "reproducibility of the IBL standardised protocol — "
            "a key feature of the dataset."
        )
        if "lab" in all_trials.columns:
            st.plotly_chart(plot_cross_subject_perf(all_trials),
                            use_container_width=True)
        else:
            st.info("Lab column not available in this dataset subset.")


# ══════════════════════════════════════════════════════════════════════
# PAGE ▸ Learning Dynamics
# ══════════════════════════════════════════════════════════════════════
elif "Learning" in page:
    st.title("Learning Dynamics & Trajectory Clustering")
    require_data()

    st.markdown(
        "We represent each subject as a **60-day performance vector** "
        "(performance on easy trials per day, interpolated/padded), then apply "
        "**K-Means clustering** on this population of vectors to identify "
        "distinct learning strategies."
    )
    st.divider()

    tab_traj, tab_cluster = st.tabs(["Individual Trajectories", "Cluster Analysis"])

    # ── Individual trajectories ───────────────────────────────────────
    with tab_traj:
        ctrl, _ = st.columns([1, 2])
        with ctrl:
            hl_subj = st.selectbox("Highlight subject", ["None"] + subject_list, key="traj_hl")
        hl = None if hl_subj == "None" else hl_subj
        st.plotly_chart(plot_learning_curves(all_trials, highlight_subject=hl),
                        use_container_width=True)

    # ── Clustering ────────────────────────────────────────────────────
    with tab_cluster:
        ctrl2, _ = st.columns([1, 2])
        with ctrl2:
            n_clust = st.slider("Number of clusters (K)", 2, 6, 3)
            run_clust = st.button("▶  Run Clustering", type="primary")

        if run_clust or "cluster_result" in st.session_state:
            if run_clust:
                with st.spinner("Clustering learning trajectories…"):
                    subject_df, km, pca_coords = cluster_learning_curves(
                        all_trials, n_clusters=n_clust
                    )
                st.session_state["cluster_result"] = (subject_df, km, pca_coords, n_clust)

            subject_df, km, pca_coords, k_used = st.session_state["cluster_result"]

            if k_used != n_clust:
                st.info(f"Showing result for K={k_used}.  Click **Run Clustering** to update.")

            st.plotly_chart(
                plot_clustered_trajectories(subject_df, pca_coords, k_used),
                use_container_width=True,
            )

            with st.expander("Cluster membership table"):
                st.dataframe(
                    subject_df[["subject", "cluster"]]
                    .sort_values("cluster")
                    .rename(columns={"cluster": "Cluster"}),
                    use_container_width=True,
                )
        else:
            st.info("Adjust K and click **Run Clustering** to start.")


# ══════════════════════════════════════════════════════════════════════
# PAGE ▸ Choice Decoder
# ══════════════════════════════════════════════════════════════════════
elif "Decoder" in page:
    st.title("🔮 Choice Decoder")
    require_data()

    st.markdown(
        """
A **logistic regression** decoder predicts the animal's choice on each trial
from four behavioral features:

| Feature | What it captures |
|---|---|
| `signed_contrast` | Bottom-up sensory evidence |
| `probabilityLeft` | Top-down block prior (internal state) |
| `prev_choice` | History-dependent bias / serial dependence |
| `prev_correct` | Win-stay / lose-shift strategy |

The relative coefficient magnitudes reveal *which internal variable drives
choice the most* — even in the absence of neural recordings.
"""
    )
    st.divider()

    ctrl, _ = st.columns([1, 2])
    with ctrl:
        sel_dec = st.selectbox("Select subject", subject_list, key="dec_sel")
        test_frac = st.slider("Test set fraction", 0.1, 0.4, 0.25, step=0.05)
        run_dec = st.button("▶  Fit Decoder", type="primary", use_container_width=True)

    if run_dec:
        subj_dec = get_subject_df(all_trials, sel_dec)
        with st.spinner("Fitting logistic regression decoder…"):
            result = fit_choice_decoder(subj_dec, test_frac=test_frac)
        if not result:
            st.error("Insufficient data for this subject (need ≥ 50 trials with all features).")
        else:
            st.session_state["decoder_result"]  = result
            st.session_state["decoder_subject"] = sel_dec

    if "decoder_result" in st.session_state:
        res   = st.session_state["decoder_result"]
        subj  = st.session_state["decoder_subject"]

        st.markdown(f"### Results for **{subj}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{res['accuracy']:.1%}")
        m2.metric("AUC",       f"{res['auc']:.3f}")
        m3.metric("Train trials", f"{res['n_train']:,}")
        m4.metric("Test trials",  f"{res['n_test']:,}")

        st.plotly_chart(plot_decoder_results(res), use_container_width=True)

        with st.expander("Interpretation"):
            st.markdown(
                """
- **`signed_contrast`** coefficient: the stimulus drives the decision — larger positive
  value means stronger bottom-up influence.
- **`probabilityLeft`** coefficient: the block prior biases choices — this is the
  *internal state* contribution, decoupled from the stimulus.
- **`prev_choice`** coefficient: positive → win-stay strategy; negative → alternation.
- **`prev_correct`** coefficient: win-stay/lose-shift tendency.

If `probabilityLeft` has a large coefficient, the animal has learned the block structure
and is using its internal state estimate to guide decisions — even on 0%-contrast trials.
"""
            )
    else:
        st.info("Select a subject and click **Fit Decoder** to start.")


# ══════════════════════════════════════════════════════════════════════
# PAGE ▸ About
# ══════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.title(" About")
    st.markdown(
        """
## IBL Brain-Wide Behavior Explorer

This application analyses decision-making behavior from the
[International Brain Laboratory](https://www.internationalbrainlab.com/) (IBL)
Brain-Wide Map dataset — a standardised 2AFC perceptual decision-making task
deployed across 12 labs using high-density Neuropixels recordings.

The behavioral analyses here focus on how **internal state** — encoded as the
hidden block prior — shapes decision-making at the population level.
This is a direct conceptual analog to the homeostatic and motivational states
studied in systems neuroscience.

---

## Analyses

| Analysis | Method | Maps to |
|---|---|---|
| Psychometric curve | 4-parameter sigmoid (curve_fit) | GLMs, signal detection |
| Block prior bias | Stratified P(right) by block condition | Internal state encoding |
| Bias per subject | Δ P(right) across block conditions | Population-level statistics |
| Learning trajectory clustering | K-Means + PCA | Unsupervised learning |
| Choice decoder | Logistic regression | Neural decoding, GLMs |
| Reaction time | Mean ± SEM by contrast | Drift-diffusion dynamics |

---

## Data

Downloaded from the IBL ONE API — public dataset, CC BY 4.0 license.

```
one.alyx.rest('datasets', 'list', tag='2021_Q1_IBL_et_al_Behaviour')
```

---

## References

- International Brain Laboratory *et al.* (2021).
  Standardised and reproducible measurement of decision-making in mice.
  **eLife**, 10:e63711.
- International Brain Laboratory *et al.* (2023).
  A Brain-Wide Map of Neural Activity during Complex Behaviour.
  **bioRxiv**.
- Ashwood Z.C. *et al.* (2022).
  Mice alternate between discrete strategies during perceptual decision-making.
  **Nature Neuroscience**, 25, 201–212.
- Findling C. *et al.* (2021).
  Imprecise neural computations as a source of adaptive behaviour in uncertain environments.
  **Nature Human Behaviour**, 5, 1036–1051.

---

## Links

- [IBL website](https://www.internationalbrainlab.com/)
- [ONE API docs](https://int-brain-lab.github.io/iblenv/)
- [Dataset paper (eLife 2021)](https://elifesciences.org/articles/63711)
- [Brain-Wide Map paper](https://www.biorxiv.org/content/10.1101/2023.07.10.548413)
"""
    )
