from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter


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

OUTPUT_CURVE = (
    PROJECT_ROOT
    / "outputs"
    / "competing_risks_cif.csv"
)

OUTPUT_HORIZONS = (
    PROJECT_ROOT
    / "outputs"
    / "competing_risks_pd_horizons.csv"
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

# Event coding from Step 06:
#
# 0 = Censor
# 1 = Default
# 2 = Voluntary Prepayment

CENSOR_EVENT = 0
DEFAULT_EVENT = 1
PREPAY_EVENT = 2

# Primary reporting horizons:
# 12, 24, 36 months
#
# 60 months is retained as a sensitivity / long-horizon
# reference, not the primary comparison horizon.

HORIZONS = [
    12,
    24,
    36,
    60,
]


# ============================================================
# 3. HELPER: EXTRACT CIF VALUE AT A HORIZON
# ============================================================

def cif_at_horizon(
    cif_series,
    horizon
):
    """
    Return the cumulative incidence at or immediately
    before a requested horizon.

    Aalen-Johansen estimates are step functions, so if
    the exact horizon is not present in the event table,
    the most recent estimate before that horizon is used.
    """

    eligible = cif_series[
        cif_series.index <= horizon
    ]

    if eligible.empty:
        return 0.0

    return float(
        eligible.iloc[-1]
    )


# ============================================================
# 4. HELPER: FIT AALEN-JOHANSEN CIF
# ============================================================

def fit_cif(
    durations,
    events,
    event_of_interest
):
    """
    Fit an Aalen-Johansen cumulative incidence function
    for one event of interest while retaining the other
    terminal event as a competing event.
    """

    fitter = AalenJohansenFitter(
        calculate_variance=False
    )

    fitter.fit(
        durations=durations,
        event_observed=events,
        event_of_interest=event_of_interest
    )

    cif = (
        fitter
        .cumulative_density_
        .iloc[:, 0]
        .copy()
    )

    cif.index = (
        pd.to_numeric(
            cif.index,
            errors="coerce"
        )
    )

    cif = (
        cif[
            cif.index.notna()
        ]
        .sort_index()
    )

    return cif


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "COMPETING RISKS CIF ANALYSIS\n"
    )

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Survival dataset not found:\n"
            f"{INPUT_FILE}"
        )

    OUTPUT_CURVE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load only required columns
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "loan_id",
            "vintage",
            "duration_months",
            "event_code",
        ]
    )

    print(
        f"Loans loaded          : "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Validate duration
    # --------------------------------------------------------

    df[
        "duration_months"
    ] = pd.to_numeric(
        df[
            "duration_months"
        ],
        errors="coerce"
    )

    missing_duration = (
        df[
            "duration_months"
        ]
        .isna()
        .sum()
    )

    negative_duration = (
        df[
            "duration_months"
        ]
        .lt(0)
        .sum()
    )

    # --------------------------------------------------------
    # Validate event codes
    # --------------------------------------------------------

    df[
        "event_code"
    ] = pd.to_numeric(
        df[
            "event_code"
        ],
        errors="coerce"
    )

    valid_event_codes = {
        CENSOR_EVENT,
        DEFAULT_EVENT,
        PREPAY_EVENT,
    }

    invalid_event_mask = (
        ~df[
            "event_code"
        ]
        .isin(
            valid_event_codes
        )
    )

    invalid_events = (
        invalid_event_mask.sum()
    )

    duplicate_loan_ids = (
        df[
            "loan_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Missing duration      : "
        f"{missing_duration:,}"
    )

    print(
        f"Negative duration     : "
        f"{negative_duration:,}"
    )

    print(
        f"Invalid event codes   : "
        f"{invalid_events:,}"
    )

    print(
        f"Duplicate loan IDs    : "
        f"{duplicate_loan_ids:,}"
    )

    if missing_duration > 0:

        raise ValueError(
            "Missing duration detected."
        )

    if negative_duration > 0:

        raise ValueError(
            "Negative duration detected."
        )

    if invalid_events > 0:

        raise ValueError(
            "Invalid event code detected."
        )

    if duplicate_loan_ids > 0:

        raise ValueError(
            "Duplicate loan IDs detected."
        )

    # --------------------------------------------------------
    # Event totals
    # --------------------------------------------------------

    event_counts = (
        df[
            "event_code"
        ]
        .value_counts()
        .sort_index()
    )

    censor_count = int(
        event_counts.get(
            CENSOR_EVENT,
            0
        )
    )

    default_count = int(
        event_counts.get(
            DEFAULT_EVENT,
            0
        )
    )

    prepay_count = int(
        event_counts.get(
            PREPAY_EVENT,
            0
        )
    )

    print(
        "\nEvent totals"
    )

    print(
        f"  Censor     : "
        f"{censor_count:,}"
    )

    print(
        f"  Default    : "
        f"{default_count:,}"
    )

    print(
        f"  Prepayment : "
        f"{prepay_count:,}"
    )

    accounting_total = (
        censor_count
        + default_count
        + prepay_count
    )

    print(
        f"  Accounting : "
        f"{accounting_total == len(df)}"
    )

    if accounting_total != len(df):

        raise ValueError(
            "Event accounting does not match "
            "the survival cohort."
        )

    # --------------------------------------------------------
    # Prepare arrays
    # --------------------------------------------------------

    durations = (
        df[
            "duration_months"
        ]
        .astype(float)
    )

    events = (
        df[
            "event_code"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # Fit Default CIF
    # --------------------------------------------------------

    print(
        "\nFitting Aalen-Johansen "
        "Default CIF..."
    )

    default_cif = fit_cif(
        durations=durations,
        events=events,
        event_of_interest=DEFAULT_EVENT
    )

    # --------------------------------------------------------
    # Fit Prepayment CIF
    # --------------------------------------------------------

    print(
        "Fitting Aalen-Johansen "
        "Prepayment CIF..."
    )

    prepay_cif = fit_cif(
        durations=durations,
        events=events,
        event_of_interest=PREPAY_EVENT
    )

    # --------------------------------------------------------
    # Create monthly CIF curve
    # --------------------------------------------------------

    max_duration = int(
        df[
            "duration_months"
        ]
        .max()
    )

    monthly_timeline = (
        pd.Index(
            range(
                0,
                max_duration + 1
            ),
            name="month"
        )
    )

    default_monthly = (
        default_cif
        .reindex(
            monthly_timeline,
            method="ffill"
        )
        .fillna(0.0)
    )

    prepay_monthly = (
        prepay_cif
        .reindex(
            monthly_timeline,
            method="ffill"
        )
        .fillna(0.0)
    )

    cif_output = pd.DataFrame(
        {
            "month":
                monthly_timeline,

            "cif_default":
                default_monthly.values,

            "cif_prepayment":
                prepay_monthly.values,
        }
    )

    cif_output[
        "event_free_probability"
    ] = (
        1.0
        - cif_output[
            "cif_default"
        ]
        - cif_output[
            "cif_prepayment"
        ]
    )

    # --------------------------------------------------------
    # Probability consistency checks
    # --------------------------------------------------------

    tolerance = 1e-10

    invalid_default_probability = (
        (
            cif_output[
                "cif_default"
            ] < -tolerance
        )
        |
        (
            cif_output[
                "cif_default"
            ] > 1 + tolerance
        )
    ).sum()

    invalid_prepay_probability = (
        (
            cif_output[
                "cif_prepayment"
            ] < -tolerance
        )
        |
        (
            cif_output[
                "cif_prepayment"
            ] > 1 + tolerance
        )
    ).sum()

    invalid_event_free = (
        cif_output[
            "event_free_probability"
        ] < -tolerance
    ).sum()

    default_nonmonotonic = (
        cif_output[
            "cif_default"
        ]
        .diff()
        .dropna()
        .lt(
            -tolerance
        )
        .sum()
    )

    prepay_nonmonotonic = (
        cif_output[
            "cif_prepayment"
        ]
        .diff()
        .dropna()
        .lt(
            -tolerance
        )
        .sum()
    )

    print(
        "\nCIF validation"
    )

    print(
        f"  Invalid Default probabilities : "
        f"{invalid_default_probability}"
    )

    print(
        f"  Invalid Prepay probabilities  : "
        f"{invalid_prepay_probability}"
    )

    print(
        f"  Negative event-free values    : "
        f"{invalid_event_free}"
    )

    print(
        f"  Default CIF decreases         : "
        f"{default_nonmonotonic}"
    )

    print(
        f"  Prepayment CIF decreases      : "
        f"{prepay_nonmonotonic}"
    )

    if (
        invalid_default_probability > 0
        or invalid_prepay_probability > 0
        or invalid_event_free > 0
        or default_nonmonotonic > 0
        or prepay_nonmonotonic > 0
    ):

        raise ValueError(
            "CIF probability validation failed."
        )

    # --------------------------------------------------------
    # Extract horizon estimates
    # --------------------------------------------------------

    horizon_rows = []

    print(
        "\n"
        + "=" * 78
    )

    print(
        "COMPETING-RISKS CUMULATIVE "
        "INCIDENCE ESTIMATES"
    )

    print(
        "=" * 78
    )

    for horizon in HORIZONS:

        pd_default = cif_at_horizon(
            default_cif,
            horizon
        )

        cif_prepay = cif_at_horizon(
            prepay_cif,
            horizon
        )

        event_free = (
            1.0
            - pd_default
            - cif_prepay
        )

        horizon_rows.append(
            {
                "horizon_months":
                    horizon,

                "pd_default":
                    pd_default,

                "pd_default_pct":
                    pd_default * 100,

                "cif_prepayment":
                    cif_prepay,

                "cif_prepayment_pct":
                    cif_prepay * 100,

                "event_free_probability":
                    event_free,

                "event_free_pct":
                    event_free * 100,
            }
        )

        label = (
            "PRIMARY"
            if horizon in {
                12,
                24,
                36,
            }
            else "SENSITIVITY"
        )

        print(
            f"\nHorizon {horizon:>2} months "
            f"[{label}]"
        )

        print(
            f"  Default CIF / PD : "
            f"{pd_default:.6f} "
            f"({pd_default * 100:.4f}%)"
        )

        print(
            f"  Prepayment CIF   : "
            f"{cif_prepay:.6f} "
            f"({cif_prepay * 100:.4f}%)"
        )

        print(
            f"  Event-free       : "
            f"{event_free:.6f} "
            f"({event_free * 100:.4f}%)"
        )

        print(
            f"  Probability sum  : "
            f"{pd_default + cif_prepay + event_free:.6f}"
        )

    # --------------------------------------------------------
    # Crude rate diagnostic
    # --------------------------------------------------------

    crude_default_rate = (
        default_count
        / len(df)
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "IMPORTANT INTERPRETATION CHECK"
    )

    print(
        "=" * 78
    )

    print(
        f"Crude lifetime default proportion: "
        f"{crude_default_rate:.6f} "
        f"({crude_default_rate * 100:.4f}%)"
    )

    print(
        "\nDo NOT interpret the crude lifetime "
        "default proportion as PD12, PD24, "
        "or PD36."
    )

    print(
        "Primary horizon PD estimates are "
        "the Default cumulative incidence "
        "from the competing-risks model."
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    cif_output.to_csv(
        OUTPUT_CURVE,
        index=False
    )

    horizon_output = pd.DataFrame(
        horizon_rows
    )

    horizon_output.to_csv(
        OUTPUT_HORIZONS,
        index=False
    )

    print(
        f"\nMonthly CIF curve saved to:\n"
        f"{OUTPUT_CURVE}"
    )

    print(
        f"\nHorizon PD estimates saved to:\n"
        f"{OUTPUT_HORIZONS}"
    )

    print(
        "\nCompeting-risks CIF analysis "
        "complete."
    )


if __name__ == "__main__":
    main()