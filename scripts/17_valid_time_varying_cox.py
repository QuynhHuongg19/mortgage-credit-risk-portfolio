from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter


# ============================================================
# MORTGAGE CREDIT RISK PORTFOLIO
# STEP 17 - VALID TIME-VARYING CAUSE-SPECIFIC COX
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

COEFFICIENT_FILE = (
    OUTPUT_DIR
    / "valid_time_varying_cox_coefficients.csv"
)

HR_FILE = (
    OUTPUT_DIR
    / "valid_time_varying_cox_hr_by_month.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "valid_time_varying_cox_audit.csv"
)


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

# Piecewise start-stop intervals.
# These boundaries cover the main 36-month analysis and
# continue beyond 36 months so later observed Default events
# are not artificially discarded.

CUTS = [
    0.0,
    3.0,
    6.0,
    12.0,
    18.0,
    24.0,
    36.0,
    48.0,
    60.0,
]

EVALUATION_MONTHS = [
    12,
    24,
    36,
    60,
]

BASE_COVARIATES = [
    "fico_10",
    "ltv_10",
    "dti_10",
    "interest_rate_1pct",
    "loan_term_12m",
]

DISPLAY_LABELS = {
    "fico_10": "FICO (+10 points)",
    "ltv_10": (
        "Original LTV "
        "(+10 percentage points)"
    ),
    "dti_10": (
        "Original DTI "
        "(+10 percentage points)"
    ),
    "interest_rate_1pct": (
        "Original interest rate "
        "(+1 percentage point)"
    ),
    "loan_term_12m": (
        "Original loan term "
        "(+12 months)"
    ),
}


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(
        "Loading loan-level survival dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = [
        "loan_id",
        "duration_months",
        "event_code",
        "fico",
        "original_ltv",
        "original_dti",
        "original_interest_rate",
        "original_loan_term",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return df


# ============================================================
# 3. PREPARE BASELINE COVARIATES
# ============================================================

def prepare_model_sample(df):

    model = df[
        [
            "loan_id",
            "duration_months",
            "event_code",
            "fico",
            "original_ltv",
            "original_dti",
            "original_interest_rate",
            "original_loan_term",
        ]
    ].copy()

    numeric_columns = [
        "duration_months",
        "event_code",
        "fico",
        "original_ltv",
        "original_dti",
        "original_interest_rate",
        "original_loan_term",
    ]

    for column in numeric_columns:
        model[column] = pd.to_numeric(
            model[column],
            errors="coerce",
        )

    total_rows = len(model)

    model = model.dropna(
        subset=numeric_columns
    ).copy()

    complete_rows = len(model)

    # --------------------------------------------------------
    # Same scaling as Step 13 / Step 15
    # --------------------------------------------------------

    model["fico_10"] = (
        model["fico"] / 10.0
    )

    model["ltv_10"] = (
        model["original_ltv"] / 10.0
    )

    model["dti_10"] = (
        model["original_dti"] / 10.0
    )

    model["interest_rate_1pct"] = (
        model["original_interest_rate"]
    )

    model["loan_term_12m"] = (
        model["original_loan_term"] / 12.0
    )

    # --------------------------------------------------------
    # Cause-specific Cox for Default
    #
    # Default = event
    # Prepayment = censoring
    # Other censoring = censoring
    # --------------------------------------------------------

    model["default_event"] = (
        model["event_code"] == 1
    ).astype("int8")

    # Cox cannot use an event at exact t=0.
    # Same estimation convention as earlier Cox models.
    zero_duration = (
        model["duration_months"] <= 0
    )

    zero_count = int(
        zero_duration.sum()
    )

    model.loc[
        zero_duration,
        "duration_months",
    ] = 0.5

    audit = {
        "total_survival_cohort": total_rows,
        "complete_case_sample": complete_rows,
        "excluded_missing": (
            total_rows - complete_rows
        ),
        "default_events": int(
            model["default_event"].sum()
        ),
        "prepayment_events": int(
            (model["event_code"] == 2).sum()
        ),
        "censored_observations": int(
            (model["event_code"] == 0).sum()
        ),
        "zero_duration_adjusted": zero_count,
    }

    return model, audit


# ============================================================
# 4. CENTER COVARIATES
# ============================================================

def center_covariates(model):

    means = {}

    for variable in BASE_COVARIATES:

        mean_value = float(
            model[variable].mean()
        )

        means[variable] = mean_value

        model[
            f"{variable}_c"
        ] = (
            model[variable]
            - mean_value
        )

    return model, means


# ============================================================
# 5. BUILD TRUE START-STOP DATA
# ============================================================

def build_start_stop(model):

    print(
        "Building start-stop risk intervals..."
    )

    pieces = []

    max_duration = float(
        model["duration_months"].max()
    )

    boundaries = CUTS.copy()

    if max_duration > boundaries[-1]:
        boundaries.append(
            max_duration + 0.001
        )

    centered_columns = [
        f"{variable}_c"
        for variable in BASE_COVARIATES
    ]

    source_columns = [
        "loan_id",
        "duration_months",
        "default_event",
    ] + centered_columns

    source = model[
        source_columns
    ].copy()

    for left, right in zip(
        boundaries[:-1],
        boundaries[1:],
    ):

        # Loan must still be under observation
        # after the interval start.

        mask = (
            source["duration_months"]
            > left
        )

        interval = source.loc[
            mask
        ].copy()

        if interval.empty:
            continue

        interval["start"] = left

        interval["stop"] = np.minimum(
            interval["duration_months"],
            right,
        )

        # Event occurs in the interval containing
        # the observed Default time.

        interval["event"] = (
            (
                interval["default_event"] == 1
            )
            &
            (
                interval["duration_months"]
                <= right
            )
        ).astype("int8")

        # ----------------------------------------------------
        # Time-varying interaction
        #
        # The time function is evaluated at interval stop.
        # Unlike Step 14, this varies across risk intervals
        # for the same loan.
        # ----------------------------------------------------

        interval[
            "analysis_time"
        ] = interval["stop"].clip(
            lower=0.5
        )

        interval[
            "log_time"
        ] = np.log(
            interval["analysis_time"]
        )

        for variable in BASE_COVARIATES:

            centered = (
                f"{variable}_c"
            )

            interaction = (
                f"{variable}_x_log_time"
            )

            interval[
                interaction
            ] = (
                interval[centered]
                * interval["log_time"]
            )

        keep = [
            "loan_id",
            "start",
            "stop",
            "event",
        ]

        keep += centered_columns

        keep += [
            f"{variable}_x_log_time"
            for variable
            in BASE_COVARIATES
        ]

        pieces.append(
            interval[keep]
        )

        print(
            f"  Interval "
            f"({left:g}, {right:g}] "
            f"rows: {len(interval):,}"
        )

        del interval
        gc.collect()

    long_df = pd.concat(
        pieces,
        ignore_index=True,
    )

    del pieces
    gc.collect()

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    invalid_intervals = int(
        (
            long_df["stop"]
            <= long_df["start"]
        ).sum()
    )

    duplicated_intervals = int(
        long_df.duplicated(
            subset=[
                "loan_id",
                "start",
                "stop",
            ]
        ).sum()
    )

    event_count = int(
        long_df["event"].sum()
    )

    return (
        long_df,
        invalid_intervals,
        duplicated_intervals,
        event_count,
    )


# ============================================================
# 6. FIT COX TIME-VARYING MODEL
# ============================================================

def fit_model(long_df):

    print()
    print(
        "Fitting CoxTimeVaryingFitter..."
    )

    model_columns = [
        "loan_id",
        "start",
        "stop",
        "event",
    ]

    model_columns += [
        f"{variable}_c"
        for variable in BASE_COVARIATES
    ]

    model_columns += [
        f"{variable}_x_log_time"
        for variable in BASE_COVARIATES
    ]

    fit_df = long_df[
        model_columns
    ].copy()

    ctv = CoxTimeVaryingFitter(
        penalizer=0.0001
    )

    start_time = time.time()

    ctv.fit(
        fit_df,
        id_col="loan_id",
        event_col="event",
        start_col="start",
        stop_col="stop",
        show_progress=True,
    )

    runtime = (
        time.time()
        - start_time
    )

    return ctv, runtime


# ============================================================
# 7. COEFFICIENT OUTPUT
# ============================================================

def build_coefficient_output(ctv):

    summary = (
        ctv.summary
        .reset_index()
        .copy()
    )

    first_column = summary.columns[0]

    summary = summary.rename(
        columns={
            first_column: "term"
        }
    )

    keep_candidates = [
        "term",
        "coef",
        "exp(coef)",
        "se(coef)",
        "coef lower 95%",
        "coef upper 95%",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "z",
        "p",
    ]

    keep = [
        column
        for column in keep_candidates
        if column in summary.columns
    ]

    return summary[keep]


# ============================================================
# 8. TIME-DEPENDENT HAZARD RATIOS
# ============================================================

def build_hr_output(ctv):

    params = ctv.params_

    rows = []

    for variable in BASE_COVARIATES:

        base_name = (
            f"{variable}_c"
        )

        interaction_name = (
            f"{variable}_x_log_time"
        )

        beta = float(
            params[base_name]
        )

        gamma = float(
            params[interaction_name]
        )

        for month in EVALUATION_MONTHS:

            log_t = np.log(
                max(month, 0.5)
            )

            log_hr = (
                beta
                + gamma * log_t
            )

            hr = np.exp(
                log_hr
            )

            rows.append(
                {
                    "variable": variable,
                    "label": (
                        DISPLAY_LABELS[
                            variable
                        ]
                    ),
                    "month": month,
                    "base_coefficient": beta,
                    "log_time_coefficient": gamma,
                    "log_hazard_ratio": log_hr,
                    "hazard_ratio": hr,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 9. VALIDATION
# ============================================================

def validate_results(
    model,
    long_df,
    hr_df,
    invalid_intervals,
    duplicated_intervals,
    event_count,
):

    errors = []

    expected_events = int(
        model["default_event"].sum()
    )

    if invalid_intervals != 0:
        errors.append(
            "Invalid start-stop intervals detected."
        )

    if duplicated_intervals != 0:
        errors.append(
            "Duplicate loan/start/stop intervals detected."
        )

    if event_count != expected_events:
        errors.append(
            "Default event count changed during "
            "start-stop expansion."
        )

    if hr_df[
        "hazard_ratio"
    ].isna().any():
        errors.append(
            "Missing time-dependent hazard ratios."
        )

    if (
        hr_df["hazard_ratio"] <= 0
    ).any():
        errors.append(
            "Non-positive hazard ratio detected."
        )

    return errors


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print()
    print(
        "MORTGAGE CREDIT RISK"
    )
    print(
        "STEP 17 - VALID TIME-VARYING "
        "CAUSE-SPECIFIC COX"
    )
    print()

    df = load_data()

    model, audit = (
        prepare_model_sample(df)
    )

    del df
    gc.collect()

    model, means = (
        center_covariates(model)
    )

    (
        long_df,
        invalid_intervals,
        duplicated_intervals,
        event_count,
    ) = build_start_stop(
        model
    )

    print()
    print("=" * 72)
    print(
        "START-STOP DATA AUDIT"
    )
    print("=" * 72)

    print(
        f"Loans                  : "
        f"{len(model):,}"
    )

    print(
        f"Long-format intervals  : "
        f"{len(long_df):,}"
    )

    print(
        f"Default events expected: "
        f"{audit['default_events']:,}"
    )

    print(
        f"Default events expanded: "
        f"{event_count:,}"
    )

    print(
        f"Invalid intervals      : "
        f"{invalid_intervals:,}"
    )

    print(
        f"Duplicate intervals    : "
        f"{duplicated_intervals:,}"
    )

    ctv, runtime = fit_model(
        long_df
    )

    coefficient_df = (
        build_coefficient_output(
            ctv
        )
    )

    hr_df = build_hr_output(
        ctv
    )

    errors = validate_results(
        model=model,
        long_df=long_df,
        hr_df=hr_df,
        invalid_intervals=(
            invalid_intervals
        ),
        duplicated_intervals=(
            duplicated_intervals
        ),
        event_count=event_count,
    )

    print()
    print("=" * 72)
    print(
        "TIME-DEPENDENT HAZARD RATIOS"
    )
    print("=" * 72)

    display = hr_df[
        [
            "label",
            "month",
            "hazard_ratio",
        ]
    ].copy()

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print(
        f"Validation errors: "
        f"{len(errors)}"
    )

    if errors:

        for error in errors:
            print(
                f"  - {error}"
            )

        raise ValueError(
            "Step 17 validation failed."
        )

    # --------------------------------------------------------
    # Audit output
    # --------------------------------------------------------

    audit_rows = [
        {
            "metric": key,
            "value": value,
        }
        for key, value
        in audit.items()
    ]

    audit_rows.extend(
        [
            {
                "metric": (
                    "long_format_intervals"
                ),
                "value": len(long_df),
            },
            {
                "metric": (
                    "invalid_intervals"
                ),
                "value": invalid_intervals,
            },
            {
                "metric": (
                    "duplicate_intervals"
                ),
                "value": duplicated_intervals,
            },
            {
                "metric": (
                    "expanded_default_events"
                ),
                "value": event_count,
            },
            {
                "metric": (
                    "model_runtime_seconds"
                ),
                "value": round(
                    runtime,
                    2,
                ),
            },
            {
                "metric": (
                    "penalizer"
                ),
                "value": 0.0001,
            },
        ]
    )

    for variable, mean_value in (
        means.items()
    ):

        audit_rows.append(
            {
                "metric": (
                    f"mean_{variable}"
                ),
                "value": mean_value,
            }
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    coefficient_df.to_csv(
        COEFFICIENT_FILE,
        index=False,
    )

    hr_df.to_csv(
        HR_FILE,
        index=False,
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("MODEL OUTPUTS")
    print("=" * 72)

    print(
        f"Coefficients : "
        f"{COEFFICIENT_FILE.name}"
    )

    print(
        f"HR by month  : "
        f"{HR_FILE.name}"
    )

    print(
        f"Audit        : "
        f"{AUDIT_FILE.name}"
    )

    print(
        f"Runtime      : "
        f"{runtime:.2f} seconds"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "- This is a cause-specific Default "
        "hazard model."
    )

    print(
        "- Prepayment remains a competing event "
        "and is censored for cause-specific Cox."
    )

    print(
        "- Covariate effects are allowed to "
        "change with analysis time."
    )

    print(
        "- Start-stop intervals replace the "
        "exploratory Step 14 implementation."
    )

    print(
        "- Hazard ratios are associations, "
        "not causal effects."
    )

    print()
    print(
        "STEP 17 COMPLETE"
    )


if __name__ == "__main__":
    main()