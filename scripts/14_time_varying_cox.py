from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


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
    / "cox_time_varying_effects.csv"
)

OUTPUT_HR = (
    PROJECT_ROOT
    / "outputs"
    / "cox_time_varying_hr_by_month.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DEFAULT_EVENT = 1

BASE_VARIABLES = [
    "fico_10",
    "ltv_10",
    "dti_10",
    "interest_rate_1pct",
    "loan_term_12m",
]

RAW_VARIABLES = [
    "fico",
    "original_ltv",
    "original_dti",
    "original_interest_rate",
    "original_loan_term",
]

LABELS = {
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

# Months at which the time-dependent HR will be reported.
REPORT_MONTHS = [
    1,
    6,
    12,
    24,
    36,
    60,
]


# ============================================================
# 3. LOAD AND PREPARE DATA
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
        "duration_months",
        "event_code",
        *RAW_VARIABLES,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


def prepare_data(df):

    work = df[
        [
            "duration_months",
            "event_code",
            *RAW_VARIABLES,
        ]
    ].copy()

    for column in work.columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce"
        )

    work = work.dropna().copy()

    work = work[
        work["duration_months"] >= 0
    ].copy()

    work["default_event"] = (
        work["event_code"]
        .eq(DEFAULT_EVENT)
        .astype(int)
    )

    # --------------------------------------------------------
    # Positive Cox time
    # --------------------------------------------------------

    work["cox_duration"] = (
        work["duration_months"]
        .astype(float)
    )

    work.loc[
        work["cox_duration"] == 0,
        "cox_duration"
    ] = 0.5

    # --------------------------------------------------------
    # Baseline covariate scaling
    # --------------------------------------------------------

    work["fico_10"] = (
        work["fico"] / 10.0
    )

    work["ltv_10"] = (
        work["original_ltv"] / 10.0
    )

    work["dti_10"] = (
        work["original_dti"] / 10.0
    )

    work["interest_rate_1pct"] = (
        work["original_interest_rate"]
    )

    work["loan_term_12m"] = (
        work["original_loan_term"] / 12.0
    )

    # --------------------------------------------------------
    # Center variables before interactions.
    #
    # Centering improves numerical stability and means the
    # baseline coefficient corresponds to the effect when
    # log(time) = 0, i.e. time = 1 month.
    # --------------------------------------------------------

    centers = {}

    for variable in BASE_VARIABLES:

        center = (
            work[variable].mean()
        )

        centers[variable] = center

        work[
            f"{variable}_centered"
        ] = (
            work[variable]
            - center
        )

    # --------------------------------------------------------
    # log(time)
    # --------------------------------------------------------

    work["log_time"] = np.log(
        work["cox_duration"]
    )

    # --------------------------------------------------------
    # Covariate x log(time) interactions
    # --------------------------------------------------------

    for variable in BASE_VARIABLES:

        work[
            f"{variable}_x_log_time"
        ] = (
            work[
                f"{variable}_centered"
            ]
            * work["log_time"]
        )

    return (
        work,
        centers
    )


# ============================================================
# 4. FIT EXTENDED COX MODEL
# ============================================================

def fit_model(work):

    centered = [
        f"{variable}_centered"
        for variable
        in BASE_VARIABLES
    ]

    interactions = [
        f"{variable}_x_log_time"
        for variable
        in BASE_VARIABLES
    ]

    model_columns = [
        "cox_duration",
        "default_event",
        *centered,
        *interactions,
    ]

    fit_df = work[
        model_columns
    ].copy()

    if fit_df.isna().any().any():

        raise ValueError(
            "Missing values remain in model data."
        )

    if (
        fit_df["cox_duration"]
        <= 0
    ).any():

        raise ValueError(
            "Non-positive Cox duration detected."
        )

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
# 5. BUILD COEFFICIENT OUTPUT
# ============================================================

def build_coefficient_output(cph):

    summary = (
        cph.summary
        .reset_index()
    )

    if "covariate" in summary.columns:

        summary = summary.rename(
            columns={
                "covariate":
                    "term"
            }
        )

    elif "index" in summary.columns:

        summary = summary.rename(
            columns={
                "index":
                    "term"
            }
        )

    rows = []

    for _, row in summary.iterrows():

        term = row["term"]

        if (
            term.endswith(
                "_x_log_time"
            )
        ):

            effect_type = (
                "TIME_INTERACTION"
            )

        else:

            effect_type = (
                "BASE_EFFECT"
            )

        rows.append(
            {
                "term":
                    term,

                "effect_type":
                    effect_type,

                "coefficient":
                    row["coef"],

                "hazard_ratio":
                    row["exp(coef)"],

                "ci_lower_95":
                    row[
                        "exp(coef) lower 95%"
                    ],

                "ci_upper_95":
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
# 6. CALCULATE TIME-DEPENDENT HR
# ============================================================

def get_coef(
    cph,
    name
):

    return float(
        cph.params_.loc[name]
    )


def get_covariance(
    cph,
    name1,
    name2
):

    return float(
        cph.variance_matrix_.loc[
            name1,
            name2
        ]
    )


def build_time_hr_output(cph):

    rows = []

    for variable in BASE_VARIABLES:

        base_term = (
            f"{variable}_centered"
        )

        interaction_term = (
            f"{variable}_x_log_time"
        )

        beta = get_coef(
            cph,
            base_term
        )

        gamma = get_coef(
            cph,
            interaction_term
        )

        var_beta = get_covariance(
            cph,
            base_term,
            base_term
        )

        var_gamma = get_covariance(
            cph,
            interaction_term,
            interaction_term
        )

        cov_beta_gamma = (
            get_covariance(
                cph,
                base_term,
                interaction_term
            )
        )

        for month in REPORT_MONTHS:

            log_t = np.log(
                float(month)
            )

            log_hr = (
                beta
                + gamma * log_t
            )

            variance = (
                var_beta
                + (
                    log_t ** 2
                    * var_gamma
                )
                + (
                    2
                    * log_t
                    * cov_beta_gamma
                )
            )

            variance = max(
                variance,
                0.0
            )

            se = np.sqrt(
                variance
            )

            lower_log = (
                log_hr
                - 1.96 * se
            )

            upper_log = (
                log_hr
                + 1.96 * se
            )

            rows.append(
                {
                    "variable":
                        variable,

                    "label":
                        LABELS[
                            variable
                        ],

                    "month":
                        month,

                    "log_hr":
                        log_hr,

                    "hazard_ratio":
                        np.exp(
                            log_hr
                        ),

                    "hr_ci_lower_95":
                        np.exp(
                            lower_log
                        ),

                    "hr_ci_upper_95":
                        np.exp(
                            upper_log
                        ),

                    "time_interaction_coefficient":
                        gamma,

                    "time_interaction_p_value":
                        float(
                            cph.summary.loc[
                                interaction_term,
                                "p"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 7. VALIDATION
# ============================================================

def validate_results(
    coefficient_output,
    hr_output
):

    errors = []

    numeric_columns = [
        "coefficient",
        "hazard_ratio",
        "ci_lower_95",
        "ci_upper_95",
        "p_value",
    ]

    for column in numeric_columns:

        if not np.isfinite(
            coefficient_output[
                column
            ]
        ).all():

            errors.append(
                f"Non-finite values in {column}"
            )

    for column in [
        "hazard_ratio",
        "hr_ci_lower_95",
        "hr_ci_upper_95",
    ]:

        if (
            hr_output[column]
            <= 0
        ).any():

            errors.append(
                f"Non-positive values in {column}"
            )

    if (
        hr_output[
            "hr_ci_lower_95"
        ]
        >
        hr_output[
            "hr_ci_upper_95"
        ]
    ).any():

        errors.append(
            "Invalid confidence interval ordering"
        )

    expected_rows = (
        len(BASE_VARIABLES)
        * len(REPORT_MONTHS)
    )

    if len(hr_output) != expected_rows:

        errors.append(
            "Unexpected number of HR rows"
        )

    return errors


# ============================================================
# 8. PRINT RESULTS
# ============================================================

def print_model_summary(
    cph,
    coefficient_output
):

    print(
        "\n"
        + "=" * 116
    )

    print(
        "EXTENDED CAUSE-SPECIFIC COX MODEL "
        "WITH COVARIATE x LOG(TIME)"
    )

    print(
        "=" * 116
    )

    print(
        f"{'Term':<48}"
        f"{'Coef':>14}"
        f"{'HR':>14}"
        f"{'p-value':>18}"
    )

    print(
        "-" * 116
    )

    for _, row in (
        coefficient_output
        .iterrows()
    ):

        print(
            f"{row['term']:<48}"
            f"{row['coefficient']:>14.6f}"
            f"{row['hazard_ratio']:>14.4f}"
            f"{row['p_value']:>18.6g}"
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


def print_time_hr(
    hr_output
):

    print(
        "\n"
        + "=" * 116
    )

    print(
        "TIME-DEPENDENT HAZARD RATIOS"
    )

    print(
        "=" * 116
    )

    for variable in BASE_VARIABLES:

        subset = (
            hr_output[
                hr_output[
                    "variable"
                ]
                == variable
            ]
        )

        label = (
            LABELS[
                variable
            ]
        )

        p_value = (
            subset[
                "time_interaction_p_value"
            ]
            .iloc[0]
        )

        print(
            f"\n{label}"
        )

        print(
            f"Time interaction p-value: "
            f"{p_value:.6g}"
        )

        print(
            f"{'Month':>8}"
            f"{'HR':>14}"
            f"{'95% CI':>28}"
        )

        for _, row in (
            subset.iterrows()
        ):

            ci = (
                f"{row['hr_ci_lower_95']:.4f}"
                f" - "
                f"{row['hr_ci_upper_95']:.4f}"
            )

            print(
                f"{int(row['month']):>8}"
                f"{row['hazard_ratio']:>14.4f}"
                f"{ci:>28}"
            )

    print(
        "\nInterpretation rule:"
    )

    print(
        "  A significant covariate x log(time) "
        "term indicates that the association "
        "with the cause-specific Default hazard "
        "changes over follow-up time."
    )

    print(
        "  HRs at 12, 24, and 36 months are "
        "therefore more informative than reporting "
        "one constant HR when the PH assumption "
        "is violated."
    )

    print(
        "  These are associations, not causal effects."
    )


# ============================================================
# 9. SAVE
# ============================================================

def save_outputs(
    coefficient_output,
    hr_output
):

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    coefficient_output.to_csv(
        OUTPUT_MODEL,
        index=False
    )

    hr_output.to_csv(
        OUTPUT_HR,
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
        "STEP 14 - TIME-VARYING COX EFFECTS\n"
    )

    print(
        "Purpose:"
    )

    print(
        "  Extend the baseline cause-specific "
        "Cox model after PH diagnostic flags."
    )

    print(
        "\nSpecification:"
    )

    print(
        "  h_D(t|X) = h_0(t) exp["
        "beta*X + gamma*X*log(t)]"
    )

    df = load_data()

    print(
        f"\nSurvival cohort loaded: "
        f"{len(df):,}"
    )

    (
        work,
        centers
    ) = prepare_data(
        df
    )

    print(
        f"Model sample       : "
        f"{len(work):,}"
    )

    print(
        f"Default events     : "
        f"{int(work['default_event'].sum()):,}"
    )

    print(
        "\nCovariate centers:"
    )

    for variable in BASE_VARIABLES:

        print(
            f"  {variable:<24}"
            f"{centers[variable]:.4f}"
        )

    (
        cph,
        fit_df
    ) = fit_model(
        work
    )

    coefficient_output = (
        build_coefficient_output(
            cph
        )
    )

    hr_output = (
        build_time_hr_output(
            cph
        )
    )

    errors = validate_results(
        coefficient_output,
        hr_output
    )

    print_model_summary(
        cph,
        coefficient_output
    )

    print_time_hr(
        hr_output
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Validation errors: "
        f"{len(errors)}"
    )

    if errors:

        for error in errors:

            print(
                f"  {error}"
            )

        raise ValueError(
            "Step 14 validation failed."
        )

    save_outputs(
        coefficient_output,
        hr_output
    )

    print(
        f"\nCoefficient output saved to:\n"
        f"{OUTPUT_MODEL}"
    )

    print(
        f"\nTime-dependent HR output saved to:\n"
        f"{OUTPUT_HR}"
    )

    print(
        "\nStep 14 complete."
    )


if __name__ == "__main__":
    main()