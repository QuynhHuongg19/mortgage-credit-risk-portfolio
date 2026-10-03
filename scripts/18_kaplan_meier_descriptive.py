from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


# ============================================================
# MORTGAGE CREDIT RISK PORTFOLIO
# STEP 18 - KAPLAN-MEIER DESCRIPTIVE BENCHMARK
#
# Purpose:
#   Compare naive Kaplan-Meier Default incidence with the
#   Aalen-Johansen competing-risk Default CIF.
#
# IMPORTANT:
#   Kaplan-Meier treats Prepayment as censoring.
#   Therefore 1 - KM(Default) is NOT the primary Default PD
#   estimator when Prepayment is a competing event.
#
# Primary PD estimator:
#   Aalen-Johansen Default CIF
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SURVIVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

AJ_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "competing_risks_pd_horizons.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

KM_CURVE_FILE = (
    OUTPUT_DIR
    / "kaplan_meier_default_curve.csv"
)

KM_HORIZON_FILE = (
    OUTPUT_DIR
    / "kaplan_meier_vs_competing_risks.csv"
)

KM_AUDIT_FILE = (
    OUTPUT_DIR
    / "kaplan_meier_descriptive_audit.csv"
)


HORIZONS = [12, 24, 36, 60]


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def find_column(df, candidates):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# LOAD SURVIVAL DATA
# ============================================================

def load_survival_data():

    require_file(SURVIVAL_FILE)

    df = pd.read_csv(
        SURVIVAL_FILE,
        low_memory=False,
    )

    duration_col = find_column(
        df,
        [
            "duration_months",
            "duration",
        ],
    )

    event_col = find_column(
        df,
        [
            "event_code",
            "event",
        ],
    )

    loan_col = find_column(
        df,
        [
            "loan_id",
            "loan_identifier",
        ],
    )

    if duration_col is None:
        raise ValueError(
            "Duration column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if event_col is None:
        raise ValueError(
            "Event-code column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    work = pd.DataFrame(
        {
            "duration_months": numeric(
                df[duration_col]
            ),
            "event_code": numeric(
                df[event_col]
            ),
        }
    )

    if loan_col is not None:

        work["loan_id"] = (
            df[loan_col]
            .astype(str)
        )

    else:

        work["loan_id"] = (
            np.arange(len(work))
            .astype(str)
        )

    return work


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_survival_data(df):

    errors = []

    if df["duration_months"].isna().any():

        errors.append(
            "Missing duration values."
        )

    if df["event_code"].isna().any():

        errors.append(
            "Missing event-code values."
        )

    if (
        df["duration_months"] < 0
    ).any():

        errors.append(
            "Negative duration detected."
        )

    valid_events = {0, 1, 2}

    actual_events = set(
        df["event_code"]
        .dropna()
        .astype(int)
        .unique()
    )

    unexpected = (
        actual_events
        - valid_events
    )

    if unexpected:

        errors.append(
            "Unexpected event codes: "
            + str(
                sorted(unexpected)
            )
        )

    if df["loan_id"].duplicated().any():

        errors.append(
            "Duplicate loan IDs detected."
        )

    return errors


# ============================================================
# KAPLAN-MEIER
# ============================================================

def fit_default_km(df):

    # --------------------------------------------------------
    # Default is the event of interest.
    #
    # Event code:
    #   0 = censor
    #   1 = Default
    #   2 = Prepayment
    #
    # For naive Default KM:
    #   Default    -> event
    #   Prepayment -> censored
    #   Other      -> censored
    #
    # This is intentionally a descriptive benchmark.
    # --------------------------------------------------------

    km_event = (
        df["event_code"]
        .eq(1)
        .astype(int)
    )

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=df[
            "duration_months"
        ],
        event_observed=km_event,
        label="Default-free KM",
    )

    survival = (
        kmf
        .survival_function_
        .reset_index()
    )

    survival.columns = [
        "month",
        "km_default_free_survival",
    ]

    survival[
        "km_naive_default_incidence"
    ] = (
        1.0
        - survival[
            "km_default_free_survival"
        ]
    )

    return kmf, survival


# ============================================================
# LOAD AALEN-JOHANSEN PD
# ============================================================

def load_aj_horizons():

    require_file(AJ_FILE)

    df = pd.read_csv(
        AJ_FILE,
        low_memory=False,
    )

    horizon_col = find_column(
        df,
        [
            "horizon_months",
            "horizon",
            "month",
        ],
    )

    default_col = find_column(
        df,
        [
            "pd_default",
            "default_cif",
            "default_pd",
            "pd",
        ],
    )

    prepay_col = find_column(
        df,
        [
            "cif_prepayment",
            "prepayment_cif",
            "prepay_cif",
        ],
    )

    if horizon_col is None:
        raise ValueError(
            "AJ horizon column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if default_col is None:
        raise ValueError(
            "AJ Default CIF column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    result = pd.DataFrame(
        {
            "horizon_months": numeric(
                df[horizon_col]
            ),
            "aj_default_cif": numeric(
                df[default_col]
            ),
        }
    )

    if prepay_col is not None:

        result[
            "aj_prepayment_cif"
        ] = numeric(
            df[prepay_col]
        )

    else:

        result[
            "aj_prepayment_cif"
        ] = np.nan

    return result


# ============================================================
# HORIZON COMPARISON
# ============================================================

def build_horizon_comparison(
    kmf,
    aj_df,
):

    rows = []

    for horizon in HORIZONS:

        km_survival = float(
            kmf.predict(horizon)
        )

        km_incidence = (
            1.0
            - km_survival
        )

        aj_match = aj_df[
            aj_df[
                "horizon_months"
            ].eq(horizon)
        ]

        if aj_match.empty:

            raise ValueError(
                f"AJ estimate missing at "
                f"{horizon} months."
            )

        aj_row = (
            aj_match.iloc[0]
        )

        aj_default = float(
            aj_row[
                "aj_default_cif"
            ]
        )

        aj_prepay = float(
            aj_row[
                "aj_prepayment_cif"
            ]
        )

        absolute_difference = (
            km_incidence
            - aj_default
        )

        if aj_default > 0:

            relative_difference = (
                absolute_difference
                / aj_default
            )

        else:

            relative_difference = np.nan

        rows.append(
            {
                "horizon_months": horizon,
                "km_default_free_survival": (
                    km_survival
                ),
                "km_naive_default_incidence": (
                    km_incidence
                ),
                "aj_default_cif": (
                    aj_default
                ),
                "aj_prepayment_cif": (
                    aj_prepay
                ),
                "km_minus_aj": (
                    absolute_difference
                ),
                "relative_difference_vs_aj": (
                    relative_difference
                ),
                "primary_pd_estimator": (
                    "AALEN_JOHANSEN_CIF"
                ),
                "km_role": (
                    "DESCRIPTIVE_BENCHMARK"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# AUDIT
# ============================================================

def build_audit(
    df,
    comparison,
):

    n_total = len(df)

    n_default = int(
        df["event_code"]
        .eq(1)
        .sum()
    )

    n_prepay = int(
        df["event_code"]
        .eq(2)
        .sum()
    )

    n_censor = int(
        df["event_code"]
        .eq(0)
        .sum()
    )

    zero_duration = int(
        df["duration_months"]
        .eq(0)
        .sum()
    )

    km_above_aj = bool(
        (
            comparison[
                "km_naive_default_incidence"
            ]
            >= comparison[
                "aj_default_cif"
            ]
            - 1e-10
        ).all()
    )

    audit = pd.DataFrame(
        [
            {
                "metric": (
                    "Total survival cohort"
                ),
                "value": n_total,
            },
            {
                "metric": (
                    "Default events"
                ),
                "value": n_default,
            },
            {
                "metric": (
                    "Prepayment events"
                ),
                "value": n_prepay,
            },
            {
                "metric": (
                    "Censored observations"
                ),
                "value": n_censor,
            },
            {
                "metric": (
                    "Zero-duration observations"
                ),
                "value": zero_duration,
            },
            {
                "metric": (
                    "KM horizons evaluated"
                ),
                "value": len(
                    comparison
                ),
            },
            {
                "metric": (
                    "Naive KM incidence >= "
                    "AJ Default CIF at all "
                    "reported horizons"
                ),
                "value": km_above_aj,
            },
        ]
    )

    return audit


# ============================================================
# VALIDATION
# ============================================================

def validate_outputs(
    curve,
    comparison,
):

    errors = []

    if curve.empty:

        errors.append(
            "KM curve is empty."
        )

    probabilities = [
        "km_default_free_survival",
        "km_naive_default_incidence",
    ]

    for column in probabilities:

        values = numeric(
            curve[column]
        )

        if (
            (values < -1e-8)
            | (values > 1 + 1e-8)
        ).any():

            errors.append(
                f"Invalid probability "
                f"in {column}."
            )

    expected_horizons = set(
        HORIZONS
    )

    actual_horizons = set(
        comparison[
            "horizon_months"
        ].astype(int)
    )

    if (
        actual_horizons
        != expected_horizons
    ):

        errors.append(
            "Horizon comparison does not "
            "contain 12, 24, 36 and 60."
        )

    for column in [
        "km_naive_default_incidence",
        "aj_default_cif",
        "aj_prepayment_cif",
    ]:

        values = numeric(
            comparison[column]
        )

        if values.isna().any():

            errors.append(
                f"Missing estimate "
                f"in {column}."
            )

    return errors


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "MORTGAGE CREDIT RISK"
    )

    print(
        "STEP 18 - KAPLAN-MEIER "
        "DESCRIPTIVE BENCHMARK"
    )

    print()

    df = load_survival_data()

    input_errors = (
        validate_survival_data(
            df
        )
    )

    if input_errors:

        print(
            "INPUT VALIDATION FAILED"
        )

        for error in input_errors:

            print(
                f"- {error}"
            )

        raise ValueError(
            "Survival input validation "
            "failed."
        )

    print("=" * 72)
    print("SURVIVAL COHORT")
    print("=" * 72)

    print(
        f"Loans       : "
        f"{len(df):,}"
    )

    print(
        f"Defaults    : "
        f"{df['event_code'].eq(1).sum():,}"
    )

    print(
        f"Prepayments : "
        f"{df['event_code'].eq(2).sum():,}"
    )

    print(
        f"Censored    : "
        f"{df['event_code'].eq(0).sum():,}"
    )

    # --------------------------------------------------------
    # Fit KM
    # --------------------------------------------------------

    kmf, curve = (
        fit_default_km(
            df
        )
    )

    # --------------------------------------------------------
    # AJ comparison
    # --------------------------------------------------------

    aj_df = (
        load_aj_horizons()
    )

    comparison = (
        build_horizon_comparison(
            kmf,
            aj_df,
        )
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit = build_audit(
        df,
        comparison,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    errors = (
        validate_outputs(
            curve,
            comparison,
        )
    )

    print()
    print("=" * 72)
    print(
        "KM VS AALEN-JOHANSEN"
    )
    print("=" * 72)

    for _, row in (
        comparison.iterrows()
    ):

        horizon = int(
            row[
                "horizon_months"
            ]
        )

        print(
            f"H{horizon:<2} | "
            f"Naive KM = "
            f"{row['km_naive_default_incidence']:.4%} | "
            f"AJ Default CIF = "
            f"{row['aj_default_cif']:.4%} | "
            f"Difference = "
            f"{row['km_minus_aj']:.4%}"
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
                f"- {error}"
            )

        raise ValueError(
            "Step 18 validation failed."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve.to_csv(
        KM_CURVE_FILE,
        index=False,
    )

    comparison.to_csv(
        KM_HORIZON_FILE,
        index=False,
    )

    audit.to_csv(
        KM_AUDIT_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(
        f"KM curve     : "
        f"{KM_CURVE_FILE.name}"
    )

    print(
        f"KM vs AJ     : "
        f"{KM_HORIZON_FILE.name}"
    )

    print(
        f"Audit        : "
        f"{KM_AUDIT_FILE.name}"
    )

    print()
    print(
        "INTERPRETATION"
    )

    print(
        "- Kaplan-Meier is retained as "
        "a descriptive benchmark."
    )

    print(
        "- In the Default KM curve, "
        "Prepayment is treated as censoring."
    )

    print(
        "- Therefore 1 - KM is not interpreted "
        "as the primary cumulative Default PD."
    )

    print(
        "- Aalen-Johansen CIF remains the "
        "primary competing-risk PD estimator."
    )

    print(
        "- Differences between KM and AJ "
        "illustrate why competing events matter."
    )

    print()
    print(
        "STEP 18 COMPLETE"
    )


if __name__ == "__main__":
    main()