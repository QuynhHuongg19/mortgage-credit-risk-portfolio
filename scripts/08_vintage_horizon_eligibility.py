from pathlib import Path
import csv
from collections import defaultdict


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "vintage_horizon_eligibility.csv"
)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

# Confirmed independently from the raw performance files.
DATA_CUTOFF = 202603

HORIZONS = [
    12,
    24,
    36,
    60,
]


# ============================================================
# 3. DATE UTILITIES
# ============================================================

def parse_period(value):
    """
    Convert YYYYMM into an integer period.

    Returns None for invalid values.
    """

    if value is None:
        return None

    value = str(value).strip()

    if (
        len(value) != 6
        or not value.isdigit()
    ):
        return None

    year = int(
        value[:4]
    )

    month = int(
        value[4:]
    )

    if not (
        1900 <= year <= 2100
        and 1 <= month <= 12
    ):
        return None

    return int(value)


def month_difference(
    start_period,
    end_period
):
    """
    Calculate elapsed calendar months
    between two YYYYMM periods.

    Examples
    --------
    201602 -> 201602 = 0
    201602 -> 201603 = 1
    201602 -> 201702 = 12
    """

    if (
        start_period is None
        or end_period is None
    ):
        return None

    start_year = (
        start_period // 100
    )

    start_month = (
        start_period % 100
    )

    end_year = (
        end_period // 100
    )

    end_month = (
        end_period % 100
    )

    return (
        (end_year - start_year) * 12
        + (end_month - start_month)
    )


# ============================================================
# 4. INITIALIZE SUMMARY
# ============================================================

def new_summary():

    result = {

        "total_loans": 0,

        "invalid_start_period": 0,

        "start_after_cutoff": 0,

        "min_start_period": None,

        "max_start_period": None,

        "min_available_months": None,

        "max_available_months": None,
    }

    for horizon in HORIZONS:

        result[
            f"eligible_{horizon}"
        ] = 0

        result[
            f"not_eligible_{horizon}"
        ] = 0

    return result


# ============================================================
# 5. UPDATE SUMMARY
# ============================================================

def update_summary(
    summary,
    start_period
):

    summary[
        "total_loans"
    ] += 1

    if start_period is None:

        summary[
            "invalid_start_period"
        ] += 1

        return

    if (
        summary[
            "min_start_period"
        ] is None
        or start_period
        < summary[
            "min_start_period"
        ]
    ):

        summary[
            "min_start_period"
        ] = start_period

    if (
        summary[
            "max_start_period"
        ] is None
        or start_period
        > summary[
            "max_start_period"
        ]
    ):

        summary[
            "max_start_period"
        ] = start_period

    available_months = (
        month_difference(
            start_period,
            DATA_CUTOFF
        )
    )

    if available_months is None:

        summary[
            "invalid_start_period"
        ] += 1

        return

    if available_months < 0:

        summary[
            "start_after_cutoff"
        ] += 1

        return

    if (
        summary[
            "min_available_months"
        ] is None
        or available_months
        < summary[
            "min_available_months"
        ]
    ):

        summary[
            "min_available_months"
        ] = available_months

    if (
        summary[
            "max_available_months"
        ] is None
        or available_months
        > summary[
            "max_available_months"
        ]
    ):

        summary[
            "max_available_months"
        ] = available_months

    for horizon in HORIZONS:

        if available_months >= horizon:

            summary[
                f"eligible_{horizon}"
            ] += 1

        else:

            summary[
                f"not_eligible_{horizon}"
            ] += 1


# ============================================================
# 6. PERCENTAGE
# ============================================================

def percentage(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "VINTAGE HORIZON ELIGIBILITY\n"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vintage_summary = defaultdict(
        new_summary
    )

    overall = new_summary()

    # --------------------------------------------------------
    # Read master survival dataset
    # --------------------------------------------------------

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        required_columns = {
            "vintage",
            "first_observed_period",
        }

        missing_columns = (
            required_columns
            - set(
                reader.fieldnames or []
            )
        )

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        for row in reader:

            vintage = (
                row[
                    "vintage"
                ]
                .strip()
            )

            start_period = parse_period(
                row[
                    "first_observed_period"
                ]
            )

            update_summary(
                vintage_summary[
                    vintage
                ],
                start_period
            )

            update_summary(
                overall,
                start_period
            )

    # --------------------------------------------------------
    # Print methodological definition
    # --------------------------------------------------------

    print(
        f"Confirmed raw performance cutoff: "
        f"{DATA_CUTOFF}"
    )

    print(
        "\nAnalysis time origin:"
    )

    print(
        "  First observed performance period"
    )

    print(
        "\nCalendar eligibility at horizon H:"
    )

    print(
        "  months(first observed period -> "
        "202603) >= H"
    )

    print(
        "\nThis measures whether the raw data window"
    )

    print(
        "could administratively support H months"
    )

    print(
        "of follow-up from the analysis time origin."
    )

    # --------------------------------------------------------
    # Print vintage table
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 112
    )

    print(
        f"{'Vintage':<10}"
        f"{'Loans':>10}"
        f"{'First Obs':>12}"
        f"{'Last Start':>12}"
        f"{'12m':>14}"
        f"{'24m':>14}"
        f"{'36m':>14}"
        f"{'60m':>14}"
    )

    print(
        "=" * 112
    )

    output_rows = []

    for vintage in sorted(
        vintage_summary.keys()
    ):

        summary = vintage_summary[
            vintage
        ]

        total = summary[
            "total_loans"
        ]

        percentages = {}

        for horizon in HORIZONS:

            eligible = summary[
                f"eligible_{horizon}"
            ]

            percentages[
                horizon
            ] = percentage(
                eligible,
                total
            )

        print(
            f"{vintage:<10}"
            f"{total:>10,}"
            f"{str(summary['min_start_period']):>12}"
            f"{str(summary['max_start_period']):>12}"
            f"{percentages[12]:>13.2f}%"
            f"{percentages[24]:>13.2f}%"
            f"{percentages[36]:>13.2f}%"
            f"{percentages[60]:>13.2f}%"
        )

        output_row = {

            "vintage":
                vintage,

            "total_loans":
                total,

            "first_observed_period_min":
                summary[
                    "min_start_period"
                ],

            "first_observed_period_max":
                summary[
                    "max_start_period"
                ],

            "min_available_months":
                summary[
                    "min_available_months"
                ],

            "max_available_months":
                summary[
                    "max_available_months"
                ],

            "invalid_start_period":
                summary[
                    "invalid_start_period"
                ],

            "start_after_cutoff":
                summary[
                    "start_after_cutoff"
                ],
        }

        for horizon in HORIZONS:

            eligible = summary[
                f"eligible_{horizon}"
            ]

            output_row[
                f"eligible_{horizon}m"
            ] = eligible

            output_row[
                f"eligible_{horizon}m_pct"
            ] = round(
                percentage(
                    eligible,
                    total
                ),
                4
            )

        output_rows.append(
            output_row
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 112
    )

    print(
        "OVERALL CALENDAR ELIGIBILITY"
    )

    print(
        "=" * 112
    )

    print(
        f"Total loans          : "
        f"{overall['total_loans']:,}"
    )

    print(
        f"Invalid start period : "
        f"{overall['invalid_start_period']:,}"
    )

    print(
        f"Start after cutoff   : "
        f"{overall['start_after_cutoff']:,}"
    )

    print(
        f"First observed period: "
        f"{overall['min_start_period']}"
    )

    print(
        f"Latest start period  : "
        f"{overall['max_start_period']}"
    )

    for horizon in HORIZONS:

        eligible = overall[
            f"eligible_{horizon}"
        ]

        total = overall[
            "total_loans"
        ]

        pct = percentage(
            eligible,
            total
        )

        print(
            f"\nHorizon {horizon:>2} months"
        )

        print(
            f"  Eligible loans : "
            f"{eligible:,} "
            f"({pct:.2f}%)"
        )

        print(
            f"  Not eligible   : "
            f"{total - eligible:,} "
            f"({100 - pct:.2f}%)"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = [

        "vintage",
        "total_loans",

        "first_observed_period_min",
        "first_observed_period_max",

        "min_available_months",
        "max_available_months",

        "invalid_start_period",
        "start_after_cutoff",
    ]

    for horizon in HORIZONS:

        fieldnames.extend(
            [
                f"eligible_{horizon}m",
                f"eligible_{horizon}m_pct",
            ]
        )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    print(
        f"\nAudit saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nVintage horizon eligibility "
        "audit complete."
    )


if __name__ == "__main__":
    main()