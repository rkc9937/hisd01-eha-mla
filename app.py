"""Streamlit dashboard for the Texas smoke-free ordinance diffusion model.

Reads the committed model outputs (coefficient table + model panel) and renders
them as an interactive web interface: coefficient plot, significance table,
odds ratios, and an explorer for the discrete-time event-history panel.

Run with:  uv run streamlit run app.py
"""

import math
import os

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
COEFFICIENTS_PATH = os.path.join(DATA_DIR, "refinement_6_eha_logit_coefficients.csv")
MODEL_PANEL_PATH = os.path.join(DATA_DIR, "refinement_6_eha_model_panel.csv")
LEARNING_MECHANISM = "proportion_state_population_local_restriction"

# Friendlier labels for the model terms.
TERM_LABELS = {
    "intercept": "Intercept",
    "ordinance_profile_population": "Ordinance-profile population",
    "municipal_population": "Municipal population (ACS)",
    "percent_25_plus_high_school_or_higher": "% age 25+ HS or higher",
    "non_hispanic_white_share": "Non-Hispanic white share",
    "per_capita_income": "Per-capita income",
    "per_capita_spending": "Per-capita spending",
    "texas_smoker_percentage": "Texas smoker %",
    LEARNING_MECHANISM: "Learning mechanism (state pop. under restriction)",
}


def label(term):
    return TERM_LABELS.get(term, term)


@st.cache_data
def load_coefficients():
    return pd.read_csv(COEFFICIENTS_PATH)


@st.cache_data
def load_panel():
    return pd.read_csv(MODEL_PANEL_PATH)


def stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


st.set_page_config(
    page_title="Texas Smoke-Free Ordinance Diffusion",
    page_icon="🚭",
    layout="wide",
)

st.title("🚭 Texas Smoke-Free Ordinance Diffusion")
st.caption(
    "Discrete-time logit hazard (event-history) model of municipal smoke-free "
    "ordinance adoption in Texas. The key regressor is a **learning / diffusion "
    "mechanism**: the share of the state population already living under a local "
    "restriction."
)

if not (os.path.exists(COEFFICIENTS_PATH) and os.path.exists(MODEL_PANEL_PATH)):
    st.error(
        "Model output CSVs not found in `data/`. Run `uv run python main.py` "
        "after building the data pipeline first."
    )
    st.stop()

coefficients = load_coefficients()
panel = load_panel()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
n_rows = len(panel)
n_municipalities = panel["Municipality"].nunique()
n_events = int(panel["adopted_this_year"].sum())
year_min, year_max = int(panel["Year"].min()), int(panel["Year"].max())

learning_row = coefficients.loc[coefficients["term"] == LEARNING_MECHANISM].iloc[0]
odds_ratio_sd = math.exp(learning_row["coefficient_standardized"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Municipality-years", f"{n_rows:,}")
c2.metric("Municipalities", f"{n_municipalities}")
c3.metric("Adoption events", f"{n_events}")
c4.metric("Risk window", f"{year_min}–{year_max}")

st.divider()

# ---------------------------------------------------------------------------
# Coefficient visualization
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Standardized coefficients")
    st.caption(
        "Effect of a one-standard-deviation change in each covariate on the "
        "log-odds of adopting in a given year. Bars show ±1.96·SE (≈95% CI)."
    )

    plot_df = coefficients.loc[coefficients["term"] != "intercept"].copy()
    plot_df["label"] = plot_df["term"].map(label)
    plot_df["lower"] = (
        plot_df["coefficient_standardized"]
        - 1.96 * plot_df["standard_error_standardized"]
    )
    plot_df["upper"] = (
        plot_df["coefficient_standardized"]
        + 1.96 * plot_df["standard_error_standardized"]
    )

    try:
        import altair as alt

        base = alt.Chart(plot_df).encode(
            y=alt.Y("label:N", sort="-x", title=None),
        )
        rule = base.mark_rule(color="#9aa0a6").encode(
            x=alt.X("lower:Q", title="Standardized coefficient (log-odds)"),
            x2="upper:Q",
        )
        zero = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="#444", strokeDash=[4, 4])
            .encode(x="x:Q")
        )
        points = base.mark_circle(size=140).encode(
            x="coefficient_standardized:Q",
            color=alt.Color(
                "is_learning_mechanism:N",
                scale=alt.Scale(
                    domain=[True, False], range=["#d93025", "#1a73e8"]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Term"),
                alt.Tooltip("coefficient_standardized:Q", title="Coef", format=".3f"),
                alt.Tooltip("standard_error_standardized:Q", title="SE", format=".3f"),
                alt.Tooltip("z:Q", format=".2f"),
                alt.Tooltip("p_value:Q", title="p", format=".2g"),
            ],
        )
        st.altair_chart((rule + zero + points).properties(height=360), use_container_width=True)
    except Exception:
        # Fallback if Altair is unavailable.
        st.bar_chart(
            plot_df.set_index("label")["coefficient_standardized"]
        )

with right:
    st.subheader("Learning mechanism")
    st.markdown(f"**{label(LEARNING_MECHANISM)}**")
    m1, m2 = st.columns(2)
    m1.metric("Std. coefficient", f"{learning_row['coefficient_standardized']:.3f}")
    m2.metric("z", f"{learning_row['z']:.2f}", f"p = {learning_row['p_value']:.1e}")
    st.metric(
        "Odds ratio (per +1 SD)",
        f"{odds_ratio_sd:.2f}×",
        help=(
            "exp(standardized coefficient). A one-SD rise in the share of the "
            "state population already covered multiplies the odds of adoption "
            "this year by this factor."
        ),
    )
    if learning_row["p_value"] < 0.05:
        st.success(
            "Strong, statistically significant diffusion effect: municipalities "
            "are far more likely to adopt once a larger share of Texans already "
            "live under local smoke-free restrictions."
        )

st.divider()

# ---------------------------------------------------------------------------
# Full coefficient table
# ---------------------------------------------------------------------------
st.subheader("Coefficient table")

table = coefficients.copy()
table["Term"] = table["term"].map(label)
table["Sig."] = table["p_value"].map(stars)
table = table[
    [
        "Term",
        "coefficient_standardized",
        "standard_error_standardized",
        "z",
        "p_value",
        "Sig.",
        "coefficient_original_scale",
    ]
].rename(
    columns={
        "coefficient_standardized": "Coef (std)",
        "standard_error_standardized": "SE (std)",
        "z": "z",
        "p_value": "p-value",
        "coefficient_original_scale": "Coef (original scale)",
    }
)

st.dataframe(
    table.style.format(
        {
            "Coef (std)": "{:.4f}",
            "SE (std)": "{:.4f}",
            "z": "{:.3f}",
            "p-value": "{:.2e}",
            "Coef (original scale)": "{:.6g}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("Significance: *** p<0.001 · ** p<0.01 · * p<0.05 · . p<0.1")

st.divider()

# ---------------------------------------------------------------------------
# Adoptions over time
# ---------------------------------------------------------------------------
st.subheader("Adoptions over time")
adoptions = (
    panel.loc[panel["adopted_this_year"] == 1]
    .groupby("Year")
    .size()
    .reindex(range(year_min, year_max + 1), fill_value=0)
)
adoptions_df = pd.DataFrame(
    {
        "Year": adoptions.index,
        "Adoptions": adoptions.values,
        "Cumulative": adoptions.cumsum().values,
    }
)
tt1, tt2 = st.columns(2)
tt1.bar_chart(adoptions_df.set_index("Year")["Adoptions"], height=260)
tt2.line_chart(adoptions_df.set_index("Year")["Cumulative"], height=260)

# ---------------------------------------------------------------------------
# Panel explorer
# ---------------------------------------------------------------------------
st.subheader("Model panel explorer")
fc1, fc2 = st.columns([1, 1])
with fc1:
    municipalities = ["(all)"] + sorted(panel["Municipality"].unique())
    chosen = st.selectbox("Municipality", municipalities)
with fc2:
    only_events = st.checkbox("Adoption-year rows only", value=False)

view = panel
if chosen != "(all)":
    view = view[view["Municipality"] == chosen]
if only_events:
    view = view[view["adopted_this_year"] == 1]

st.dataframe(view, use_container_width=True, hide_index=True, height=320)
st.caption(f"{len(view):,} of {n_rows:,} municipality-year rows shown.")
