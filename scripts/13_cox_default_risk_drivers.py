from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "cox_default_risk_drivers.csv"
)

OUTPUT_SAMPLE = (
    PROJECT_ROOT
    / "outputs"
    / "cox_model_sample_audit.csv"
)

OUTPUT_PH = (
    PROJECT_ROOT
    / "outputs"
    / "cox_ph_assumption_test.csv"
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

DEFAULT_EVENT = 1

RAW_COVARIATES = [
    "fico",
    "original_ltv",
    "original_dti",
    "original_interest_rate",
    "original_loan_term",
]

MODEL_COVARIATES = [
    "fico_10",
    "ltv_10",
    "dti_10",
    "interest_rate_1pct",
    "loan_term_12m",
]

VARIABLE_LABELS = {
    "fico_10":
        "FICO (+10 points)",

    "ltv_10":
        "Original LTV (+10 percentage points)",

    "dti_10":
        "Original DTI (+10 percentage points)",

    "interest_rate_1pct":
        "Original interest rate (+1 percentage point)",

    "loan_term_12m":
        "Original loan term (+12 months)",
}


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    required = {
        "loan_id",
        "duration_months",
        "event_code",
        *RAW_COVARIATES,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return df


# ============================================================
# 4. PREPARE MODEL SAMPLE
# ============================================================

def prepare_model_data(df):

    total_loans = len(df)

    work = df[
        [
            "loan_id",
            "duration_months",
            "event_code",
            *RAW_COVARIATES,
        ]
    ].copy()

    numeric_columns = [
        "duration_months",
        "event_code",
        *RAW_COVARIATES,
    ]

    for column in numeric_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce"
        )

    missing_counts = (
        work[
            [
                "duration_months",
                "event_code",
                *RAW_COVARIATES,
            ]
        ]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # Cause-specific Default event indicator
    # --------------------------------------------------------

    work["default_event"] = (
        work["event_code"]
        .eq(DEFAULT_EVENT)
        .astype(int)
    )

    # --------------------------------------------------------
    # Complete-case sample
    # --------------------------------------------------------

    complete_columns = [
        "duration_months",
        "event_code",
        *RAW_COVARIATES,
    ]

    model_df = (
        work
        .dropna(
            subset=complete_columns
        )
        .copy()
    )

    # Survival time cannot be negative.

    model_df = model_df[
        model_df[
            "duration_months"
        ] >= 0
    ].copy()

    # --------------------------------------------------------
    # Cox PH requires positive analysis time.
    #
    # Our survival clock legitimately permits duration = 0
    # for an event/censoring in the first observed month.
    #
    # Rather than deleting these observations, use a small
    # positive offset solely for Cox model estimation.
    #
    # Original duration_months remains unchanged elsewhere.
    # --------------------------------------------------------

    zero_duration = (
        model_df[
            "duration_months"
        ] == 0
    )

    zero_duration_count = int(
        zero_duration.sum()
    )

    model_df[
        "cox_duration"
    ] = model_df[
        "duration_months"
    ].astype(float)

    model_df.loc[
        zero_duration,
        "cox_duration"
    ] = 0.5

    # --------------------------------------------------------
    # Scale covariates for interpretable HRs
    # --------------------------------------------------------

    model_df["fico_10"] = (
        model_df["fico"]
        / 10.0
    )

    model_df["ltv_10"] = (
        model_df["original_ltv"]
        / 10.0
    )

    model_df["dti_10"] = (
        model_df["original_dti"]
        / 10.0
    )

    model_df[
        "interest_rate_1pct"
    ] = (
        model_df[
            "original_interest_rate"
        ]
    )

    model_df["loan_term_12m"] = (
        model_df[
            "original_loan_term"
        ]
        / 12.0
    )

    # --------------------------------------------------------
    # Basic plausibility checks
    # These are audit flags, NOT filtering rules.
    # --------------------------------------------------------

    plausibility = {

        "fico_outside_300_850":
            int(
                (
                    (
                        model_df["fico"]
                        < 300
                    )
                    |
                    (
                        model_df["fico"]
                        > 850
                    )
                ).sum()
            ),

        "ltv_outside_0_200":
            int(
                (
                    (
                        model_df[
                            "original_ltv"
                        ]
                        < 0
                    )
                    |
                    (
                        model_df[
                            "original_ltv"
                        ]
                        > 200
                    )
                ).sum()
            ),

        "dti_outside_0_100":
            int(
                (
                    (
                        model_df[
                            "original_dti"
                        ]
                        < 0
                    )
                    |
                    (
                        model_df[
                            "original_dti"
                        ]
                        > 100
                    )
                ).sum()
            ),

        "interest_rate_nonpositive":
            int(
                (
                    model_df[
                        "original_interest_rate"
                    ]
                    <= 0
                ).sum()
            ),

        "loan_term_nonpositive":
            int(
                (
                    model_df[
                        "original_loan_term"
                    ]
                    <= 0
                ).sum()
            ),
    }

    # --------------------------------------------------------
    # Sample audit table
    # --------------------------------------------------------

    audit_rows = []

    audit_rows.append(
        {
            "metric":
                "Total survival cohort",

            "value":
                total_loans,
        }
    )

    audit_rows.append(
        {
            "metric":
                "Complete-case Cox sample",

            "value":
                len(model_df),
        }
    )

    audit_rows.append(
        {
            "metric":
                "Excluded from complete-case model",

            "value":
                total_loans
                - len(model_df),
        }
    )

    audit_rows.append(
        {
            "metric":
                "Default events in Cox sample",

            "value":
                int(
                    model_df[
                        "default_event"
                    ].sum()
                ),
        }
    )

    audit_rows.append(
        {
            "metric":
                "Non-default observations in Cox sample",

            "value":
                int(
                    (
                        model_df[
                            "default_event"
                        ]
                        == 0
                    ).sum()
                ),
        }
    )

    audit_rows.append(
        {
            "metric":
                "Zero-duration observations adjusted to 0.5",

            "value":
                zero_duration_count,
        }
    )

    for column in RAW_COVARIATES:

        audit_rows.append(
            {
                "metric":
                    f"Missing {column}",

                "value":
                    int(
                        missing_counts[
                            column
                        ]
                    ),
            }
        )

    for metric, value in (
        plausibility.items()
    ):

        audit_rows.append(
            {
                "metric":
                    metric,

                "value":
                    value,
            }
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    return (
        model_df,
        audit_df
    )


# ============================================================
# 5. FIT CAUSE-SPECIFIC COX MODEL
# ============================================================

def fit_cox_model(model_df):

    fit_df = model_df[
        [
            "cox_duration",
            "default_event",
            *MODEL_COVARIATES,
        ]
    ].copy()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if fit_df.isna().any().any():

        raise ValueError(
            "Missing values remain "
            "in Cox model data."
        )

    if (
        fit_df[
            "cox_duration"
        ] <= 0
    ).any():

        raise ValueError(
            "Cox duration must be > 0."
        )

    if (
        fit_df[
            "default_event"
        ].sum()
        == 0
    ):

        raise ValueError(
            "No Default events available."
        )

    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    cph = CoxPHFitter()

    with warnings.catch_warnings():

        warnings.simplefilter(
            "default"
        )

        cph.fit(
            fit_df,
            duration_col="cox_duration",
            event_col="default_event",
            show_progress=False,
        )

    return (
        cph,
        fit_df
    )


# ============================================================
# 6. MODEL RESULTS
# ============================================================

def build_model_output(cph):

    summary = (
        cph.summary
        .reset_index()
        .rename(
            columns={
                "covariate":
                    "variable"
            }
        )
    )

    rows = []

    for _, row in (
        summary.iterrows()
    ):

        variable = row[
            "variable"
        ]

        rows.append(
            {
                "variable":
                    variable,

                "label":
                    VARIABLE_LABELS.get(
                        variable,
                        variable
                    ),

                "coefficient":
                    row["coef"],

                "hazard_ratio":
                    row["exp(coef)"],

                "hr_ci_lower_95":
                    row[
                        "exp(coef) lower 95%"
                    ],

                "hr_ci_upper_95":
                    row[
                        "exp(coef) upper 95%"
                    ],

                "standard_error":
                    row["se(coef)"],

                "z_statistic":
                    row["z"],

                "p_value":
                    row["p"],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 7. PROPORTIONAL HAZARDS TEST
# ============================================================

def test_ph_assumption(
    cph,
    fit_df
):

    result = (
        proportional_hazard_test(
            cph,
            fit_df,
            time_transform="rank",
        )
    )

    ph_df = (
        result.summary
        .reset_index()
        .rename(
            columns={
                "index":
                    "variable",

                "test_statistic":
                    "ph_test_statistic",

                "p":
                    "ph_p_value",
            }
        )
    )

    if (
        "variable"
        not in ph_df.columns
    ):

        first_column = (
            ph_df.columns[0]
        )

        ph_df = ph_df.rename(
            columns={
                first_column:
                    "variable"
            }
        )

    ph_df[
        "ph_assumption_flag"
    ] = np.where(
        ph_df[
            "ph_p_value"
        ] < 0.05,
        "POTENTIAL_VIOLATION",
        "NO_EVIDENCE_AT_5PCT",
    )

    return ph_df


# ============================================================
# 8. PRINT RESULTS
# ============================================================

def print_audit(
    audit_df
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        "COX MODEL SAMPLE AUDIT"
    )

    print(
        "=" * 78
    )

    for _, row in (
        audit_df.iterrows()
    ):

        print(
            f"{row['metric']:<52}"
            f"{int(row['value']):>15,}"
        )


def print_model_results(
    model_output,
    cph
):

    print(
        "\n"
        + "=" * 112
    )

    print(
        "CAUSE-SPECIFIC COX MODEL - "
        "DEFAULT RISK DRIVERS"
    )

    print(
        "=" * 112
    )

    print(
        f"{'Variable':<48}"
        f"{'HR':>12}"
        f"{'95% CI':>24}"
        f"{'p-value':>16}"
    )

    print(
        "-" * 112
    )

    for _, row in (
        model_output.iterrows()
    ):

        ci = (
            f"{row['hr_ci_lower_95']:.4f}"
            f" - "
            f"{row['hr_ci_upper_95']:.4f}"
        )

        print(
            f"{row['label']:<48}"
            f"{row['hazard_ratio']:>12.4f}"
            f"{ci:>24}"
            f"{row['p_value']:>16.6g}"
        )

    print(
        "\nModel diagnostics:"
    )

    print(
        f"  Observations : "
        f"{cph._n_examples:,}"
    )

    print(
        f"  Concordance  : "
        f"{cph.concordance_index_:.4f}"
    )

    print(
        f"  Partial AIC  : "
        f"{cph.AIC_partial_:.2f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  HR > 1 indicates a higher "
        "cause-specific Default hazard."
    )

    print(
        "  HR < 1 indicates a lower "
        "cause-specific Default hazard."
    )

    print(
        "  Associations are conditional on "
        "the other covariates in this model."
    )

    print(
        "  They should not be interpreted "
        "as causal effects."
    )


def print_ph_results(
    ph_df
):

    print(
        "\n"
        + "=" * 92
    )

    print(
        "PROPORTIONAL HAZARDS ASSUMPTION TEST"
    )

    print(
        "=" * 92
    )

    print(
        f"{'Variable':<28}"
        f"{'Statistic':>16}"
        f"{'p-value':>16}"
        f"{'Flag':>28}"
    )

    print(
        "-" * 92
    )

    for _, row in (
        ph_df.iterrows()
    ):

        print(
            f"{row['variable']:<28}"
            f"{row['ph_test_statistic']:>16.4f}"
            f"{row['ph_p_value']:>16.6g}"
            f"{row['ph_assumption_flag']:>28}"
        )

    violations = (
        ph_df[
            "ph_assumption_flag"
        ]
        .eq(
            "POTENTIAL_VIOLATION"
        )
        .sum()
    )

    print(
        f"\nPotential PH violations "
        f"at p < 0.05: "
        f"{violations}"
    )

    if violations > 0:

        print(
            "Do not ignore these variables."
        )

        print(
            "They will be reviewed before "
            "final interpretation and may "
            "require time-varying effects, "
            "stratification, or sensitivity "
            "analysis."
        )


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    model_output,
    audit_df,
    ph_df
):

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_output.to_csv(
        OUTPUT_MODEL,
        index=False
    )

    audit_df.to_csv(
        OUTPUT_SAMPLE,
        index=False
    )

    ph_df.to_csv(
        OUTPUT_PH,
        index=False
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK"
    )

    print(
        "STEP 13 - COX DEFAULT "
        "RISK DRIVERS\n"
    )

    print(
        "Model:"
    )

    print(
        "  Cause-specific Cox PH "
        "for Default"
    )

    print(
        "\nEvent definition:"
    )

    print(
        "  Default = first numeric "
        "delinquency status >= 3"
    )

    print(
        "\nCompeting event treatment:"
    )

    print(
        "  Voluntary Prepayment is censored "
        "at its observed event time "
        "for this cause-specific Cox model."
    )

    print(
        "\nCovariates:"
    )

    for variable in MODEL_COVARIATES:

        print(
            f"  {VARIABLE_LABELS[variable]}"
        )

    df = load_data()

    print(
        f"\nSurvival cohort loaded: "
        f"{len(df):,}"
    )

    (
        model_df,
        audit_df
    ) = prepare_model_data(
        df
    )

    print_audit(
        audit_df
    )

    (
        cph,
        fit_df
    ) = fit_cox_model(
        model_df
    )

    model_output = (
        build_model_output(
            cph
        )
    )

    print_model_results(
        model_output,
        cph
    )

    print(
        "\nRunning proportional "
        "hazards diagnostic..."
    )

    ph_df = (
        test_ph_assumption(
            cph,
            fit_df
        )
    )

    print_ph_results(
        ph_df
    )

    save_outputs(
        model_output,
        audit_df,
        ph_df
    )

    print(
        f"\nModel results saved to:\n"
        f"{OUTPUT_MODEL}"
    )

    print(
        f"\nSample audit saved to:\n"
        f"{OUTPUT_SAMPLE}"
    )

    print(
        f"\nPH diagnostic saved to:\n"
        f"{OUTPUT_PH}"
    )

    print(
        "\nStep 13 complete."
    )


if __name__ == "__main__":
    main()