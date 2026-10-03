from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mortgage Credit Risk Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

FILES = {
    "summary": OUTPUT / "dashboard_summary.csv",
    "vintage": OUTPUT / "dashboard_vintage_pd.csv",
    "risk": OUTPUT / "dashboard_risk_drivers.csv",
    "shock": OUTPUT / "dashboard_shock_analysis.csv",
    "km": OUTPUT / "kaplan_meier_vs_competing_risks.csv",
    "km_curve": OUTPUT / "kaplan_meier_default_curve.csv",
}


# ============================================================
# COLOR THEME
# ============================================================

PINK = "#E75480"
PINK_DARK = "#B83268"
PINK_LIGHT = "#FCE7F0"
ROSE = "#F28AAE"
PURPLE = "#8D5A97"
ORANGE = "#D99A4E"
TEXT = "#2B2430"
MUTED = "#746A73"
GRID = "#EFE5EA"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                180deg,
                #fffafd 0%,
                #ffffff 28%,
                #fff9fc 100%
            );
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #fff1f7 0%,
                #fffafd 100%
            );
        border-right: 1px solid #f3dce6;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #b83268 0%,
                #e75480 55%,
                #f28aae 100%
            );
        box-shadow:
            0 14px 35px rgba(184, 50, 104, 0.18);
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin: 0;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 16px;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #3a2932;
        margin-top: 12px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #746a73;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .info-box {
        background: #fff4f8;
        border: 1px solid #f4cfde;
        border-left: 5px solid #e75480;
        border-radius: 12px;
        padding: 14px 17px;
        color: #4c3943;
        margin-top: 8px;
        margin-bottom: 18px;
    }

    .warning-box {
        background: #fff9f2;
        border: 1px solid #f1dfc7;
        border-left: 5px solid #d99a4e;
        border-radius: 12px;
        padding: 14px 17px;
        color: #594735;
        margin-top: 8px;
        margin-bottom: 18px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid #f2dce5;
        padding: 17px 18px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(140,70,100,0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #746a73;
    }

    div[data-testid="stMetricValue"] {
        color: #b83268;
    }

    hr {
        border-color: #f0dce5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(
        path,
        low_memory=False,
    )


def require_files():

    missing = [
        str(path)
        for path in FILES.values()
        if not path.exists()
    ]

    if missing:

        st.error(
            "Missing required dashboard files:\n\n"
            + "\n".join(missing)
        )

        st.stop()


require_files()

summary = load_csv(FILES["summary"])
vintage = load_csv(FILES["vintage"])
risk = load_csv(FILES["risk"])
shock = load_csv(FILES["shock"])
km = load_csv(FILES["km"])
km_curve = load_csv(FILES["km_curve"])


# ============================================================
# HELPERS
# ============================================================

def pct(x, digits=2):

    if pd.isna(x):
        return "N/A"

    return (
        f"{float(x) * 100:.{digits}f}%"
    )


def chart_layout(
    fig,
    height=430,
):

    fig.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color=TEXT,
        ),
        title_font=dict(
            size=18,
            color=TEXT,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
    )

    return fig


def section(
    title,
    subtitle=None,
):

    st.markdown(
        f'<div class="section-title">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:

        st.markdown(
            f'<div class="section-subtitle">'
            f'{subtitle}</div>',
            unsafe_allow_html=True,
        )


def existing_columns(
    df,
    columns,
):

    return [
        column
        for column in columns
        if column in df.columns
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🏠 Mortgage Risk"
    )

    st.caption(
        "Survival & Competing Risks Analytics"
    )

    st.markdown("---")

    page = st.radio(
        "Explore",
        [
            "Executive Overview",
            "Competing Risks",
            "Vintage Analysis",
            "Risk Drivers",
            "Macroeconomic Shocks",
            "Methodology",
        ],
    )

    st.markdown("---")

    st.markdown(
        "### Analysis scope"
    )

    st.write(
        "**Freddie Mac Single-Family "
        "Loan-Level Dataset**"
    )

    st.write(
        "Origination vintages: "
        "**2016–2026**"
    )

    st.write(
        "Primary horizon: "
        "**36 months**"
    )

    st.write(
        "Sensitivity horizon: "
        "**60 months**"
    )

    st.markdown("---")

    st.caption(
        "Dashboard reads validated analytical "
        "outputs. Models are not refitted at runtime."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Mortgage Credit Risk Analytics</h1>
        <p>
            Default timing, competing prepayment risk,
            vintage heterogeneity and macroeconomic exposure
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    section(
        "Portfolio Risk Overview",
        (
            "Primary cumulative Default risk is estimated "
            "with Aalen–Johansen competing-risk CIF."
        ),
    )

    def pd_at(horizon):

        value = summary.loc[
            summary[
                "horizon_months"
            ].eq(horizon),
            "default_cif",
        ]

        return float(
            value.iloc[0]
        )

    pd12 = pd_at(12)
    pd24 = pd_at(24)
    pd36 = pd_at(36)
    pd60 = pd_at(60)

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "12-Month PD",
        pct(pd12),
    )

    c2.metric(
        "24-Month PD",
        pct(pd24),
    )

    c3.metric(
        "36-Month PD",
        pct(pd36),
    )

    c4.metric(
        "60-Month PD",
        pct(pd60),
        help=(
            "Sensitivity analysis only."
        ),
    )

    st.markdown("")

    left, right = st.columns(
        [1.45, 1]
    )

    with left:

        plot = summary.copy()

        plot[
            "Default CIF (%)"
        ] = (
            plot[
                "default_cif"
            ] * 100
        )

        plot[
            "Prepayment CIF (%)"
        ] = (
            plot[
                "prepayment_cif"
            ] * 100
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=plot[
                    "horizon_months"
                ],
                y=plot[
                    "Default CIF (%)"
                ],
                mode="lines+markers",
                name="Default CIF",
                line=dict(
                    color=PINK_DARK,
                    width=4,
                ),
                marker=dict(
                    size=9
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=plot[
                    "horizon_months"
                ],
                y=plot[
                    "Prepayment CIF (%)"
                ],
                mode="lines+markers",
                name="Prepayment CIF",
                line=dict(
                    color=PURPLE,
                    width=3,
                ),
                marker=dict(
                    size=8
                ),
            )
        )

        fig.update_layout(
            title=(
                "Cumulative Incidence "
                "by Horizon"
            ),
            xaxis_title=(
                "Loan age / elapsed months"
            ),
            yaxis_title=(
                "Cumulative incidence (%)"
            ),
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True,
        )

    with right:

        st.markdown(
            """
            <div class="info-box">
            <b>Why 36 months?</b><br><br>

            Administrative eligibility is approximately
            <b>69.62%</b> at 36 months compared with
            <b>50.29%</b> at 60 months.

            <br><br>

            The 36-month horizon also supports a substantially
            more mature vintage comparison. Therefore PD36 is
            the primary medium-term horizon, while PD60 is
            retained as sensitivity analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="warning-box">
            <b>Vintage comparability</b><br><br>

            The primary mature PD36 comparison uses
            <b>2016–2022</b> vintages.

            Vintages <b>2023–2026</b> do not have comparable
            36-month administrative maturity and are
            explicitly flagged.
            </div>
            """,
            unsafe_allow_html=True,
        )

    section(
        "Core Analytical Framework",
        (
            "Different models answer different credit-risk "
            "questions and should not be interpreted as "
            "interchangeable estimates."
        ),
    )

    framework = pd.DataFrame(
        {
            "Method": [
                "Aalen–Johansen",
                "Kaplan–Meier",
                "Cause-Specific Cox",
                "Start–Stop Cox",
                "Fine–Gray",
            ],
            "Role": [
                (
                    "Primary cumulative "
                    "Default PD"
                ),
                (
                    "Descriptive benchmark"
                ),
                (
                    "Instantaneous Default "
                    "hazard drivers"
                ),
                (
                    "Time-varying "
                    "hazard effects"
                ),
                (
                    "Default CIF-oriented "
                    "sensitivity model"
                ),
            ],
        }
    )

    st.dataframe(
        framework,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# COMPETING RISKS
# ============================================================

elif page == "Competing Risks":

    section(
        "Default vs Voluntary Prepayment",
        (
            "Prepayment removes a loan from future Default "
            "risk and is therefore modeled as a competing event."
        ),
    )

    comparison = km.copy()

    comparison[
        "KM Default (%)"
    ] = (
        comparison[
            "km_naive_default_incidence"
        ] * 100
    )

    comparison[
        "AJ Default CIF (%)"
    ] = (
        comparison[
            "aj_default_cif"
        ] * 100
    )

    comparison[
        "Difference (pp)"
    ] = (
        comparison[
            "km_minus_aj"
        ] * 100
    )

    c1, c2 = st.columns(
        [1.4, 1]
    )

    with c1:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=comparison[
                    "horizon_months"
                ],
                y=comparison[
                    "KM Default (%)"
                ],
                mode="lines+markers",
                name="1 − Kaplan–Meier",
                line=dict(
                    color=ROSE,
                    width=3,
                    dash="dash",
                ),
                marker=dict(
                    size=8
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=comparison[
                    "horizon_months"
                ],
                y=comparison[
                    "AJ Default CIF (%)"
                ],
                mode="lines+markers",
                name="Aalen–Johansen CIF",
                line=dict(
                    color=PINK_DARK,
                    width=4,
                ),
                marker=dict(
                    size=9
                ),
            )
        )

        fig.update_layout(
            title=(
                "Naive KM vs "
                "Competing-Risk Default CIF"
            ),
            xaxis_title=(
                "Horizon (months)"
            ),
            yaxis_title=(
                "Default incidence (%)"
            ),
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True,
        )

    with c2:

        for _, row in (
            comparison.iterrows()
        ):

            st.metric(
                (
                    f"{int(row['horizon_months'])}"
                    "-Month Gap"
                ),
                (
                    f"{row['Difference (pp)']:.3f} pp"
                ),
                help=(
                    "Naive KM Default incidence "
                    "minus Aalen–Johansen Default CIF."
                ),
            )

    st.markdown(
        """
        <div class="info-box">
        Kaplan–Meier treats voluntary Prepayment as censoring.

        Under competing risks,
        <b>1 − KM is not the Default CIF</b>.

        The widening gap over time illustrates why
        Aalen–Johansen is used as the primary cumulative
        Default PD estimator in this project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    display = comparison[
        [
            "horizon_months",
            "KM Default (%)",
            "AJ Default CIF (%)",
            "Difference (pp)",
        ]
    ].copy()

    display.columns = [
        "Horizon",
        "Naive KM Default (%)",
        "AJ Default CIF (%)",
        "KM − AJ (pp)",
    ]

    st.dataframe(
        display.round(4),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# VINTAGE ANALYSIS
# ============================================================

elif page == "Vintage Analysis":

    section(
        "Mortgage Vintage Comparison",
        (
            "Cumulative Default risk is compared at the same "
            "loan age rather than using raw lifetime Default rates."
        ),
    )

    plot = vintage.copy()

    plot[
        "vintage"
    ] = (
        plot[
            "vintage"
        ]
        .astype(int)
        .astype(str)
    )

    long = plot.melt(
        id_vars=[
            "vintage",
            "pd36_status",
        ],
        value_vars=[
            "pd12",
            "pd24",
            "pd36",
        ],
        var_name="Horizon",
        value_name="PD",
    )

    long[
        "PD (%)"
    ] = (
        long["PD"] * 100
    )

    long[
        "Horizon"
    ] = (
        long[
            "Horizon"
        ].map(
            {
                "pd12": "PD12",
                "pd24": "PD24",
                "pd36": "PD36",
            }
        )
    )

    fig = px.line(
        long,
        x="vintage",
        y="PD (%)",
        color="Horizon",
        markers=True,
        title=(
            "Default CIF by "
            "Origination Vintage"
        ),
        color_discrete_sequence=[
            ROSE,
            PINK_DARK,
            PURPLE,
        ],
    )

    fig.update_layout(
        xaxis_title=(
            "Origination vintage"
        ),
        yaxis_title=(
            "Default CIF (%)"
        ),
    )

    st.plotly_chart(
        chart_layout(
            fig,
            480,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="warning-box">
        <b>
        Do not read 2023–2026 PD36 as fully mature
        vintage outcomes.
        </b>

        <br><br>

        The primary 36-month vintage comparison uses
        2016–2022.

        Later vintages are displayed for transparency
        but are maturity-limited.
        </div>
        """,
        unsafe_allow_html=True,
    )

    table = vintage[
        [
            "vintage",
            "pd12",
            "pd24",
            "pd36",
            "pd36_status",
        ]
    ].copy()

    for column in [
        "pd12",
        "pd24",
        "pd36",
    ]:

        table[
            column
        ] = (
            table[
                column
            ] * 100
        ).round(3)

    table.columns = [
        "Vintage",
        "PD12 (%)",
        "PD24 (%)",
        "PD36 (%)",
        "PD36 Status",
    ]

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# RISK DRIVERS
# ============================================================

elif page == "Risk Drivers":

    section(
        "Default Risk Drivers",
        (
            "Hazard ratios describe associations with Default "
            "risk; they are not causal effects."
        ),
    )

    labels = {
        "fico_10": "FICO +10",
        "ltv_10": "Original LTV +10pp",
        "dti_10": "Original DTI +10pp",
        "interest_rate_1pct": (
            "Interest Rate +1pp"
        ),
        "loan_term_12m": (
            "Loan Term +12m"
        ),
    }

    r = risk.copy()

    r[
        "Driver"
    ] = (
        r[
            "variable"
        ]
        .map(labels)
        .fillna(
            r["variable"]
        )
    )

    baseline = r[
        [
            "Driver",
            "cause_specific_hr",
            "fine_gray_shr",
        ]
    ].copy()

    baseline.columns = [
        "Driver",
        "Cause-Specific Cox",
        "Fine–Gray",
    ]

    long = baseline.melt(
        id_vars="Driver",
        var_name="Model",
        value_name="Hazard Ratio",
    )

    fig = px.bar(
        long,
        x="Hazard Ratio",
        y="Driver",
        color="Model",
        barmode="group",
        orientation="h",
        title=(
            "Baseline Risk Driver Estimates"
        ),
        color_discrete_sequence=[
            PINK_DARK,
            PURPLE,
        ],
    )

    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color="#777777",
    )

    st.plotly_chart(
        chart_layout(
            fig,
            470,
        ),
        use_container_width=True,
    )

    st.caption(
        "Cause-specific HR and Fine–Gray sHR answer "
        "different questions and should not be treated "
        "as numerically interchangeable."
    )

    section(
        "How Effects Change with Loan Age",
        (
            "Validated Step 17 start–stop cause-specific Cox "
            "allows covariate effects to vary with analysis time."
        ),
    )

    tv_columns = {
        "time_varying_hr_12m": 12,
        "time_varying_hr_24m": 24,
        "time_varying_hr_36m": 36,
        "time_varying_hr_60m": 60,
    }

    rows = []

    for _, row in r.iterrows():

        for (
            column,
            month,
        ) in tv_columns.items():

            rows.append(
                {
                    "Driver": (
                        row[
                            "Driver"
                        ]
                    ),
                    "Month": month,
                    "Hazard Ratio": (
                        row[
                            column
                        ]
                    ),
                }
            )

    tv = pd.DataFrame(
        rows
    )

    selected_driver = (
        st.selectbox(
            "Select a risk driver",
            list(
                labels.values()
            ),
        )
    )

    selected = tv[
        tv[
            "Driver"
        ].eq(
            selected_driver
        )
    ]

    fig = px.line(
        selected,
        x="Month",
        y="Hazard Ratio",
        markers=True,
        title=(
            "Time-Varying Effect: "
            f"{selected_driver}"
        ),
        color_discrete_sequence=[
            PINK_DARK
        ],
    )

    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="#777777",
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Step 17 supersedes the exploratory Step 14 model.

        The dashboard uses the validated start–stop
        risk-set formulation for time-varying effects.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MACROECONOMIC SHOCKS
# ============================================================

elif page == "Macroeconomic Shocks":

    section(
        "Macroeconomic Shock Exposure",
        (
            "The analysis examines whether loans' early-life "
            "observation windows overlap selected macroeconomic "
            "episodes and how loan behavior differs across periods."
        ),
    )

    st.markdown(
        """
        <div class="warning-box">
        Results on this page are
        <b>descriptive associations</b>.

        They do not establish that a macroeconomic episode
        caused the observed Default, delinquency or
        Prepayment behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )

    exposure = shock[
        shock[
            "analysis_type"
        ].eq(
            "FIRST_36_MONTH_EXPOSURE"
        )
    ].copy()

    period = shock[
        shock[
            "analysis_type"
        ].eq(
            "BEFORE_DURING_AFTER"
        )
    ].copy()

    # --------------------------------------------------------
    # FIRST 36-MONTH EXPOSURE
    # --------------------------------------------------------

    section(
        "First 36-Month Exposure",
        (
            "Loans are classified by whether their first "
            "36 observed months overlap the selected shock windows."
        ),
    )

    exposure_columns = existing_columns(
        exposure,
        [
            "shock_group",
            "loans",
            "default_within_36_pct",
            "prepay_within_36_pct",
            "ever_30dpd_first36_pct",
            "ever_60dpd_first36_pct",
            "ever_90dpd_first36_pct",
        ],
    )

    exposure_clean = (
        exposure[
            exposure_columns
        ].copy()
    )

    rename_exposure = {
        "shock_group": "Exposure Group",
        "loans": "Loans",
        "default_within_36_pct": (
            "Default ≤36m (%)"
        ),
        "prepay_within_36_pct": (
            "Prepayment ≤36m (%)"
        ),
        "ever_30dpd_first36_pct": (
            "Ever 30+ DPD (%)"
        ),
        "ever_60dpd_first36_pct": (
            "Ever 60+ DPD (%)"
        ),
        "ever_90dpd_first36_pct": (
            "Ever 90+ DPD (%)"
        ),
    }

    exposure_clean = (
        exposure_clean.rename(
            columns=rename_exposure
        )
    )

    exposure_clean[
        "Exposure Group"
    ] = (
        exposure_clean[
            "Exposure Group"
        ]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
    )

    numeric_exposure = [
        column
        for column in exposure_clean.columns
        if "(%)" in column
    ]

    for column in numeric_exposure:

        exposure_clean[
            column
        ] = pd.to_numeric(
            exposure_clean[
                column
            ],
            errors="coerce",
        )

    if (
        "Default ≤36m (%)"
        in exposure_clean.columns
        and
        "Prepayment ≤36m (%)"
        in exposure_clean.columns
    ):

        exposure_plot = (
            exposure_clean[
                [
                    "Exposure Group",
                    "Default ≤36m (%)",
                    "Prepayment ≤36m (%)",
                ]
            ]
            .melt(
                id_vars=(
                    "Exposure Group"
                ),
                var_name="Outcome",
                value_name="Rate (%)",
            )
        )

        fig = px.bar(
            exposure_plot,
            x="Exposure Group",
            y="Rate (%)",
            color="Outcome",
            barmode="group",
            title=(
                "36-Month Outcomes "
                "by Shock Exposure"
            ),
            color_discrete_sequence=[
                PINK_DARK,
                PURPLE,
            ],
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Loans (%)",
        )

        st.plotly_chart(
            chart_layout(
                fig,
                460,
            ),
            use_container_width=True,
        )

    st.dataframe(
        exposure_clean.round(3),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Exposure groups indicate whether a loan's first "
        "36 observed months overlap the selected shock windows. "
        "These percentages are descriptive cohort outcomes."
    )

    # --------------------------------------------------------
    # BEFORE / DURING / AFTER
    # --------------------------------------------------------

    section(
        "Before / During / After",
        (
            "Monthly loan behavior is compared across equal-length "
            "windows around each selected macroeconomic episode."
        ),
    )

    period_candidates = [
        "shock",
        "shock_name",
        "episode",
        "period",
        "phase",
        "window",
        "start_period",
        "end_period",
        "observed_loan_months",
        "loan_months",
        "delinq_30_rate",
        "delinq_30_rate_pct",
        "dpd30_rate",
        "dpd30_rate_pct",
        "delinq_60_rate",
        "delinq_60_rate_pct",
        "dpd60_rate",
        "dpd60_rate_pct",
        "delinq_90_rate",
        "delinq_90_rate_pct",
        "dpd90_rate",
        "dpd90_rate_pct",
        "default_rate",
        "default_rate_pct",
        "prepayment_rate",
        "prepayment_rate_pct",
        "prepay_rate",
        "prepay_rate_pct",
    ]

    period_columns = (
        existing_columns(
            period,
            period_candidates,
        )
    )

    # Remove columns that are completely empty
    period_clean = (
        period[
            period_columns
        ].copy()
        if period_columns
        else pd.DataFrame()
    )

    period_clean = (
        period_clean.dropna(
            axis=1,
            how="all",
        )
    )

    # If schema names differ, automatically retain every
    # non-empty column from BEFORE_DURING_AFTER rows.
    if period_clean.empty:

        period_clean = (
            period.drop(
                columns=[
                    "analysis_type",
                    "interpretation_scope",
                    "causal_interpretation_allowed",
                ],
                errors="ignore",
            )
            .dropna(
                axis=1,
                how="all",
            )
            .copy()
        )

    else:

        additional_nonempty = (
            period.drop(
                columns=[
                    "analysis_type",
                    "interpretation_scope",
                    "causal_interpretation_allowed",
                ],
                errors="ignore",
            )
            .dropna(
                axis=1,
                how="all",
            )
        )

        for column in (
            additional_nonempty.columns
        ):

            if (
                column
                not in period_clean.columns
            ):

                period_clean[
                    column
                ] = (
                    additional_nonempty[
                        column
                    ]
                )

    # Friendly labels for known Step 12 fields
    rename_period = {
        "shock": "Shock",
        "shock_name": "Shock",
        "episode": "Shock",
        "period": "Phase",
        "phase": "Phase",
        "window": "Phase",
        "start_period": "Start",
        "end_period": "End",
        "observed_loan_months": (
            "Observed Loan-Months"
        ),
        "loan_months": (
            "Observed Loan-Months"
        ),
        "delinq_30_rate": (
            "30+ DPD Rate"
        ),
        "delinq_30_rate_pct": (
            "30+ DPD Rate (%)"
        ),
        "dpd30_rate": (
            "30+ DPD Rate"
        ),
        "dpd30_rate_pct": (
            "30+ DPD Rate (%)"
        ),
        "delinq_60_rate": (
            "60+ DPD Rate"
        ),
        "delinq_60_rate_pct": (
            "60+ DPD Rate (%)"
        ),
        "dpd60_rate": (
            "60+ DPD Rate"
        ),
        "dpd60_rate_pct": (
            "60+ DPD Rate (%)"
        ),
        "delinq_90_rate": (
            "90+ DPD Rate"
        ),
        "delinq_90_rate_pct": (
            "90+ DPD Rate (%)"
        ),
        "dpd90_rate": (
            "90+ DPD Rate"
        ),
        "dpd90_rate_pct": (
            "90+ DPD Rate (%)"
        ),
        "default_rate": (
            "Default Rate"
        ),
        "default_rate_pct": (
            "Default Rate (%)"
        ),
        "prepayment_rate": (
            "Prepayment Rate"
        ),
        "prepayment_rate_pct": (
            "Prepayment Rate (%)"
        ),
        "prepay_rate": (
            "Prepayment Rate"
        ),
        "prepay_rate_pct": (
            "Prepayment Rate (%)"
        ),
    }

    period_clean = (
        period_clean.rename(
            columns=rename_period
        )
    )

    # Detect phase / shock columns after renaming
    shock_col = (
        "Shock"
        if "Shock" in period_clean.columns
        else None
    )

    phase_col = (
        "Phase"
        if "Phase" in period_clean.columns
        else None
    )

    # Candidate rate columns
    rate_columns = [
        column
        for column in period_clean.columns
        if (
            "Rate" in str(column)
            or "rate" in str(column)
        )
    ]

    # Build a chart if identifiable fields exist
    if (
        phase_col is not None
        and rate_columns
    ):

        preferred = [
            column
            for column in rate_columns
            if (
                "Default" in column
                or "Prepayment" in column
                or "90+" in column
            )
        ]

        if not preferred:

            preferred = (
                rate_columns[:3]
            )

        chart_id = [
            phase_col
        ]

        if shock_col is not None:
            chart_id.insert(
                0,
                shock_col,
            )

        chart_data = (
            period_clean[
                chart_id
                + preferred
            ].copy()
        )

        for column in preferred:

            chart_data[
                column
            ] = pd.to_numeric(
                chart_data[
                    column
                ],
                errors="coerce",
            )

        chart_long = (
            chart_data.melt(
                id_vars=chart_id,
                value_vars=preferred,
                var_name="Outcome",
                value_name="Rate",
            )
        )

        # If raw proportions were supplied,
        # convert to percentage.
        nonmissing = (
            chart_long[
                "Rate"
            ].dropna()
        )

        if (
            not nonmissing.empty
            and
            nonmissing.max() <= 1
        ):

            chart_long[
                "Rate"
            ] = (
                chart_long[
                    "Rate"
                ] * 100
            )

        if shock_col is not None:

            shock_options = (
                chart_long[
                    shock_col
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_shock = (
                st.selectbox(
                    "Select macroeconomic episode",
                    shock_options,
                )
            )

            chart_long = (
                chart_long[
                    chart_long[
                        shock_col
                    ].astype(str).eq(
                        selected_shock
                    )
                ]
            )

        fig = px.line(
            chart_long,
            x=phase_col,
            y="Rate",
            color="Outcome",
            markers=True,
            title=(
                "Loan-Month Event Rates "
                "Across Shock Phases"
            ),
            color_discrete_sequence=[
                PINK_DARK,
                PURPLE,
                ROSE,
                ORANGE,
            ],
        )

        fig.update_layout(
            xaxis_title="Shock phase",
            yaxis_title=(
                "Observed loan-month rate (%)"
            ),
        )

        st.plotly_chart(
            chart_layout(
                fig,
                460,
            ),
            use_container_width=True,
        )

    st.dataframe(
        period_clean,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Interpretation rule</b><br><br>

        Before / During / After event rates use
        <b>observed loan-months</b> as the denominator.

        They are therefore measures of monthly behavior,
        not cumulative PD and not Aalen–Johansen CIF.

        Differences across periods are descriptive and
        should not be interpreted as causal estimates.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    section(
        "Research Design",
        (
            "The project treats mortgage Default as a "
            "time-to-event problem with voluntary Prepayment "
            "as a competing terminal event."
        ),
    )

    st.markdown(
        "### Event framework"
    )

    st.markdown(
        """
        **Default:** first observed month with numeric
        delinquency status of 3 or greater, representing
        90+ days delinquent under the project definition.

        **Voluntary Prepayment:** Freddie Mac Zero Balance
        Code 01 occurring before Default.

        **Censoring:** end of observation or selected
        non-study termination events occurring before
        Default or Prepayment.

        **Same-month Default and Prepayment:** assigned to
        Default as a documented project convention.
        """
    )

    st.markdown(
        "### Time scale"
    )

    st.markdown(
        """
        Survival time is measured as calendar-month elapsed
        time from the loan's first observed performance
        record to its event or censoring month.

        Freddie Mac `LOAN AGE` is retained for audit and
        context but is not used as the primary survival
        duration because sequence auditing identified a
        small number of inconsistencies.
        """
    )

    st.markdown(
        "### Survival framework"
    )

    st.write(
        "Survival function:"
    )

    st.latex(
        r"S(t)=P(T>t)"
    )

    st.write(
        "Cause-specific Default hazard:"
    )

    st.latex(
        r"""
        h_D(t)
        =
        \lim_{\Delta t\to0}
        \frac{
        P(t\leq T<t+\Delta t,\ J=D\mid T\geq t)
        }{\Delta t}
        """
    )

    st.write(
        "With Default and Prepayment as terminal events:"
    )

    st.latex(
        r"""
        S(t)
        =
        \exp\left[
        -\int_0^t
        \left(
        h_D(u)+h_P(u)
        \right)du
        \right]
        """
    )

    st.markdown(
        "### Primary competing-risk estimator"
    )

    st.write(
        "For Default event D:"
    )

    st.latex(
        r"""
        CIF_D(t)
        =
        P(T\leq t,\ J=D)
        """
    )

    st.latex(
        r"""
        F_D(t)
        =
        \int_0^t
        S(u^-)\,h_D(u)\,du
        """
    )

    st.write(
        "For voluntary Prepayment event P:"
    )

    st.latex(
        r"""
        CIF_P(t)
        =
        P(T\leq t,\ J=P)
        """
    )

    st.latex(
        r"""
        F_P(t)
        =
        \int_0^t
        S(u^-)\,h_P(u)\,du
        """
    )

    st.markdown(
        """
        The cumulative incidence functions are estimated
        using the **Aalen–Johansen estimator**.
        """
    )

    st.markdown(
        "### Kaplan–Meier benchmark"
    )

    st.latex(
        r"""
        \widehat{S}_{KM}(t)
        =
        \prod_{t_i\leq t}
        \left(
        1-\frac{d_i}{n_i}
        \right)
        """
    )

    st.markdown(
        """
        Kaplan–Meier is retained as a descriptive benchmark.

        When Prepayment is a competing event:
        """
    )

    st.latex(
        r"""
        1-\widehat{S}_{KM,D}(t)
        \neq
        \widehat{F}_D(t)
        """
    )

    st.markdown(
        """
        Therefore **1 − KM is not interpreted as the primary
        cumulative Default PD** in this project.
        """
    )

    st.markdown(
        "### Cause-specific Cox model"
    )

    st.latex(
        r"""
        h(t\mid X)
        =
        h_0(t)
        \exp(\beta^\top X)
        """
    )

    st.latex(
        r"""
        HR_j
        =
        \exp(\beta_j)
        """
    )

    st.markdown(
        """
        The cause-specific Cox model evaluates the
        instantaneous Default hazard among loans that
        remain event-free.

        Voluntary Prepayment is treated as a competing
        termination and is censored at its occurrence for
        the Default cause-specific hazard model.
        """
    )

    st.markdown(
        "### Time-varying Cox model"
    )

    st.latex(
        r"""
        h(t\mid X(t))
        =
        h_0(t)
        \exp\left(
        \beta^\top X(t)
        \right)
        """
    )

    st.markdown(
        """
        The final dashboard uses the validated **Step 17
        start–stop risk-set formulation**.

        This supersedes the exploratory Step 14
        implementation.
        """
    )

    st.markdown(
        "### Fine–Gray model"
    )

    st.markdown(
        """
        Fine–Gray provides a complementary
        **subdistribution-hazard** view linked to the
        Default cumulative incidence function.

        Cause-specific Cox and Fine–Gray therefore answer
        different questions and their hazard ratios should
        not be interpreted as interchangeable.

        Full-cohort Fine–Gray estimates in this portfolio
        are **point estimates only**. Variance estimation
        was disabled, so confidence intervals and p-values
        are not reported.
        """
    )

    st.markdown(
        "### Primary PD horizons"
    )

    st.markdown(
        """
        **PD12, PD24 and PD36** are primary analytical
        horizons.

        **PD60** is retained as sensitivity analysis.
        """
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "36-Month Administrative Eligibility",
        "69.62%",
    )

    c2.metric(
        "60-Month Administrative Eligibility",
        "50.29%",
    )

    st.markdown(
        """
        The 36-month horizon provides stronger follow-up
        coverage and supports a mature vintage comparison
        through the 2022 origination cohort.

        At 60 months, administrative maturity deteriorates
        substantially for newer vintages.

        The project therefore uses:

        **PD12 / PD24 / PD36 → primary analysis**

        **PD60 → sensitivity analysis**
        """
    )

    st.markdown(
        "### Vintage interpretation"
    )

    st.markdown(
        """
        The primary mature PD36 comparison uses
        **2016–2022** origination vintages.

        **2023–2026** are displayed for transparency but
        are classified as **maturity-limited** at the
        36-month horizon.

        The 95% maturity threshold used for primary vintage
        reporting is a **project reporting convention**,
        not a Freddie Mac or Aalen–Johansen standard.
        """
    )

    st.markdown(
        "### Macroeconomic analysis"
    )

    st.markdown(
        """
        The project evaluates whether the first 36 observed
        months of each loan overlap selected macroeconomic
        episodes.

        It also compares delinquency, Default and
        Prepayment behavior across Before / During / After
        windows.

        These comparisons are **descriptive associations**.
        They do not identify causal effects.
        """
    )

    section(
        "Validated Analytical Outputs",
    )

    status = pd.DataFrame(
        {
            "Component": [
                (
                    "Data ingestion & "
                    "schema audit"
                ),
                "Event chronology",
                "Survival dataset",
                "Aalen–Johansen CIF",
                "Vintage analysis",
                "Baseline Cox",
                (
                    "Start–stop "
                    "time-varying Cox"
                ),
                "Fine–Gray",
                (
                    "KM descriptive "
                    "benchmark"
                ),
                (
                    "Macroeconomic "
                    "shock analysis"
                ),
            ],
            "Status": [
                "Validated",
                "Validated",
                "Validated",
                "Validated",
                "Validated",
                "Validated",
                "Validated",
                (
                    "Point estimates "
                    "validated"
                ),
                "Validated",
                "Descriptive",
            ],
        }
    )

    st.dataframe(
        status,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Mortgage Credit Risk Portfolio • "
    "Freddie Mac loan-level survival & "
    "competing-risks analysis"
)