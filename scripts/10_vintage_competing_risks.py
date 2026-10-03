from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SURVIVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

ELIGIBILITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "vintage_horizon_eligibility.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "vintage_competing_risks_pd.csv"
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

CENSOR_EVENT = 0
DEFAULT_EVENT = 1
PREPAY_EVENT = 2

HORIZONS = [
    12,
    24,
    36,
]

# Reporting rule:
# A vintage-horizon estimate is treated as fully comparable
# only when at least 95% of loans in that vintage had the
# administrative opportunity to reach that horizon.
#
# Estimates below this threshold are still calculated, but
# must be interpreted as maturity-limited estimates.

FULL_COMPARABILITY_THRESHOLD = 95.0


# ============================================================
# 3. AALEN-JOHANSEN FUNCTIONS
# ============================================================

def fit_cif(
    durations,
    events,
    event_of_interest
):
    """
    Fit the Aalen-Johansen cumulative incidence function
    for one event of interest.
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

    cif.index = pd.to_numeric(
        cif.index,
        errors="coerce"
    )

    cif = (
        cif[
            cif.index.notna()
        ]
        .sort_index()
    )

    return cif


def cif_at_horizon(
    cif_series,
    horizon
):
    """
    Return CIF at or immediately before horizon.
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
# 4. REPORTING STATUS
# ============================================================

def reporting_status(
    eligibility_pct
):
    """
    Classify whether a vintage-horizon estimate has
    sufficient administrative maturity for direct
    cross-vintage comparison.
    """

    if pd.isna(
        eligibility_pct
    ):
        return "ELIGIBILITY_UNKNOWN"

    if (
        eligibility_pct
        >= FULL_COMPARABILITY_THRESHOLD
    ):
        return "FULLY_COMPARABLE"

    return "MATURITY_LIMITED"


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "VINTAGE COMPETING RISKS ANALYSIS\n"
    )

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not SURVIVAL_FILE.exists():

        raise FileNotFoundError(
            f"Survival dataset not found:\n"
            f"{SURVIVAL_FILE}"
        )

    if not ELIGIBILITY_FILE.exists():

        raise FileNotFoundError(
            f"Eligibility audit not found:\n"
            f"{ELIGIBILITY_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load survival data
    # --------------------------------------------------------

    df = pd.read_csv(
        SURVIVAL_FILE,
        usecols=[
            "loan_id",
            "vintage",
            "duration_months",
            "event_code",
        ]
    )

    df[
        "duration_months"
    ] = pd.to_numeric(
        df[
            "duration_months"
        ],
        errors="coerce"
    )

    df[
        "event_code"
    ] = pd.to_numeric(
        df[
            "event_code"
        ],
        errors="coerce"
    )

    df[
        "vintage"
    ] = pd.to_numeric(
        df[
            "vintage"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Validate survival data
    # --------------------------------------------------------

    invalid_duration = (
        df[
            "duration_months"
        ]
        .isna()
        |
        df[
            "duration_months"
        ]
        .lt(0)
    )

    valid_codes = {
        CENSOR_EVENT,
        DEFAULT_EVENT,
        PREPAY_EVENT,
    }

    invalid_event = (
        ~df[
            "event_code"
        ]
        .isin(
            valid_codes
        )
    )

    invalid_vintage = (
        df[
            "vintage"
        ]
        .isna()
    )

    duplicate_loans = (
        df[
            "loan_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Loans loaded          : "
        f"{len(df):,}"
    )

    print(
        f"Invalid duration      : "
        f"{invalid_duration.sum():,}"
    )

    print(
        f"Invalid event code    : "
        f"{invalid_event.sum():,}"
    )

    print(
        f"Invalid vintage       : "
        f"{invalid_vintage.sum():,}"
    )

    print(
        f"Duplicate loan IDs    : "
        f"{duplicate_loans:,}"
    )

    if invalid_duration.any():

        raise ValueError(
            "Invalid duration detected."
        )

    if invalid_event.any():

        raise ValueError(
            "Invalid event code detected."
        )

    if invalid_vintage.any():

        raise ValueError(
            "Invalid vintage detected."
        )

    if duplicate_loans > 0:

        raise ValueError(
            "Duplicate loan IDs detected."
        )

    df[
        "vintage"
    ] = (
        df[
            "vintage"
        ]
        .astype(int)
    )

    df[
        "event_code"
    ] = (
        df[
            "event_code"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # Load calendar eligibility from Step 08
    # --------------------------------------------------------

    eligibility = pd.read_csv(
        ELIGIBILITY_FILE
    )

    eligibility[
        "vintage"
    ] = pd.to_numeric(
        eligibility[
            "vintage"
        ],
        errors="coerce"
    )

    if (
        eligibility[
            "vintage"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Invalid vintage in eligibility file."
        )

    eligibility[
        "vintage"
    ] = (
        eligibility[
            "vintage"
        ]
        .astype(int)
    )

    required_eligibility_columns = {
        "vintage",
        "total_loans",
        "eligible_12m_pct",
        "eligible_24m_pct",
        "eligible_36m_pct",
    }

    missing_columns = (
        required_eligibility_columns
        - set(
            eligibility.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Eligibility file missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    eligibility = (
        eligibility
        .set_index(
            "vintage"
        )
    )

    # --------------------------------------------------------
    # Vintage accounting
    # --------------------------------------------------------

    vintage_counts = (
        df[
            "vintage"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nVintage cohort sizes:"
    )

    for vintage, count in (
        vintage_counts.items()
    ):

        print(
            f"  {vintage}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Run Aalen-Johansen by vintage
    # --------------------------------------------------------

    output_rows = []

    vintages = sorted(
        df[
            "vintage"
        ]
        .unique()
    )

    for vintage in vintages:

        vintage_df = (
            df[
                df[
                    "vintage"
                ]
                == vintage
            ]
            .copy()
        )

        total_loans = len(
            vintage_df
        )

        default_events = int(
            (
                vintage_df[
                    "event_code"
                ]
                == DEFAULT_EVENT
            )
            .sum()
        )

        prepay_events = int(
            (
                vintage_df[
                    "event_code"
                ]
                == PREPAY_EVENT
            )
            .sum()
        )

        censored = int(
            (
                vintage_df[
                    "event_code"
                ]
                == CENSOR_EVENT
            )
            .sum()
        )

        print(
            "\n"
            + "-" * 78
        )

        print(
            f"Vintage {vintage}"
        )

        print(
            f"  Loans      : "
            f"{total_loans:,}"
        )

        print(
            f"  Default    : "
            f"{default_events:,}"
        )

        print(
            f"  Prepayment : "
            f"{prepay_events:,}"
        )

        print(
            f"  Censor     : "
            f"{censored:,}"
        )

        durations = (
            vintage_df[
                "duration_months"
            ]
            .astype(float)
        )

        events = (
            vintage_df[
                "event_code"
            ]
            .astype(int)
        )

        default_cif = fit_cif(
            durations=durations,
            events=events,
            event_of_interest=DEFAULT_EVENT
        )

        prepay_cif = fit_cif(
            durations=durations,
            events=events,
            event_of_interest=PREPAY_EVENT
        )

        # ----------------------------------------------------
        # Horizon estimates
        # ----------------------------------------------------

        for horizon in HORIZONS:

            default_pd = cif_at_horizon(
                default_cif,
                horizon
            )

            prepay_probability = (
                cif_at_horizon(
                    prepay_cif,
                    horizon
                )
            )

            event_free = (
                1.0
                - default_pd
                - prepay_probability
            )

            eligibility_column = (
                f"eligible_{horizon}m_pct"
            )

            if (
                vintage
                in eligibility.index
            ):

                eligibility_pct = float(
                    eligibility.loc[
                        vintage,
                        eligibility_column
                    ]
                )

            else:

                eligibility_pct = np.nan

            status = reporting_status(
                eligibility_pct
            )

            probability_sum = (
                default_pd
                + prepay_probability
                + event_free
            )

            if not np.isclose(
                probability_sum,
                1.0,
                atol=1e-8
            ):

                raise ValueError(
                    f"Probability accounting failed "
                    f"for vintage {vintage}, "
                    f"horizon {horizon}."
                )

            if event_free < -1e-10:

                raise ValueError(
                    f"Negative event-free probability "
                    f"for vintage {vintage}, "
                    f"horizon {horizon}."
                )

            output_rows.append(
                {
                    "vintage":
                        vintage,

                    "horizon_months":
                        horizon,

                    "total_loans":
                        total_loans,

                    "default_events_lifetime":
                        default_events,

                    "prepayment_events_lifetime":
                        prepay_events,

                    "censored_lifetime":
                        censored,

                    "calendar_eligibility_pct":
                        eligibility_pct,

                    "reporting_status":
                        status,

                    "pd_default":
                        default_pd,

                    "pd_default_pct":
                        default_pd * 100,

                    "cif_prepayment":
                        prepay_probability,

                    "cif_prepayment_pct":
                        prepay_probability * 100,

                    "event_free_probability":
                        event_free,

                    "event_free_pct":
                        event_free * 100,
                }
            )

            print(
                f"\n  Horizon {horizon:>2}m"
            )

            print(
                f"    Eligibility : "
                f"{eligibility_pct:>6.2f}%"
            )

            print(
                f"    Status      : "
                f"{status}"
            )

            print(
                f"    Default CIF : "
                f"{default_pd:.6f} "
                f"({default_pd * 100:.4f}%)"
            )

            print(
                f"    Prepay CIF  : "
                f"{prepay_probability:.6f} "
                f"({prepay_probability * 100:.4f}%)"
            )

            print(
                f"    Event-free  : "
                f"{event_free:.6f} "
                f"({event_free * 100:.4f}%)"
            )

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    result = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_rows = (
        len(vintages)
        * len(HORIZONS)
    )

    if len(result) != expected_rows:

        raise ValueError(
            "Unexpected number of "
            "vintage-horizon rows."
        )

    invalid_probabilities = (
        (
            result[
                "pd_default"
            ] < -1e-10
        )
        |
        (
            result[
                "pd_default"
            ] > 1 + 1e-10
        )
        |
        (
            result[
                "cif_prepayment"
            ] < -1e-10
        )
        |
        (
            result[
                "cif_prepayment"
            ] > 1 + 1e-10
        )
        |
        (
            result[
                "event_free_probability"
            ] < -1e-10
        )
    ).sum()

    if invalid_probabilities > 0:

        raise ValueError(
            "Invalid probabilities detected "
            "in final output."
        )

    # --------------------------------------------------------
    # Compact PD table
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 94
    )

    print(
        "VINTAGE DEFAULT CIF / PD SUMMARY"
    )

    print(
        "=" * 94
    )

    print(
        f"{'Vintage':<10}"
        f"{'PD12':>12}"
        f"{'PD24':>12}"
        f"{'PD36':>12}"
        f"{'36m Elig.':>14}"
        f"{'36m Status':>22}"
    )

    print(
        "-" * 94
    )

    for vintage in vintages:

        vintage_result = (
            result[
                result[
                    "vintage"
                ]
                == vintage
            ]
            .set_index(
                "horizon_months"
            )
        )

        pd12 = (
            vintage_result
            .loc[
                12,
                "pd_default_pct"
            ]
        )

        pd24 = (
            vintage_result
            .loc[
                24,
                "pd_default_pct"
            ]
        )

        pd36 = (
            vintage_result
            .loc[
                36,
                "pd_default_pct"
            ]
        )

        eligibility36 = (
            vintage_result
            .loc[
                36,
                "calendar_eligibility_pct"
            ]
        )

        status36 = (
            vintage_result
            .loc[
                36,
                "reporting_status"
            ]
        )

        print(
            f"{vintage:<10}"
            f"{pd12:>11.4f}%"
            f"{pd24:>11.4f}%"
            f"{pd36:>11.4f}%"
            f"{eligibility36:>13.2f}%"
            f"{status36:>22}"
        )

    # --------------------------------------------------------
    # Explicit maturity warning
    # --------------------------------------------------------

    limited_36 = (
        result[
            (
                result[
                    "horizon_months"
                ]
                == 36
            )
            &
            (
                result[
                    "reporting_status"
                ]
                != "FULLY_COMPARABLE"
            )
        ]
        [
            "vintage"
        ]
        .tolist()
    )

    print(
        "\n"
        + "=" * 94
    )

    print(
        "36-MONTH COMPARABILITY CHECK"
    )

    print(
        "=" * 94
    )

    print(
        "Rule: FULLY_COMPARABLE requires "
        f">= {FULL_COMPARABILITY_THRESHOLD:.0f}% "
        "calendar eligibility."
    )

    print(
        "\nMaturity-limited vintages at 36m:"
    )

    if limited_36:

        print(
            "  "
            + ", ".join(
                str(v)
                for v in limited_36
            )
        )

    else:

        print(
            "  None"
        )

    print(
        "\nThese CIF estimates are retained "
        "for transparency, but should not be "
        "used as directly comparable mature "
        "PD36 estimates."
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nVintage competing-risks "
        f"results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nVintage competing-risks "
        "analysis complete."
    )


if __name__ == "__main__":
    main()