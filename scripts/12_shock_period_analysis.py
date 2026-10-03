from pathlib import Path
import csv
import zipfile
from collections import defaultdict


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

SURVIVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "shock_period_analysis.csv"
)


# ============================================================
# 2. EVENT DEFINITIONS
# ============================================================

DEFAULT_EVENT = 1
PREPAY_EVENT = 2

FIRST_36_MONTHS = 36


# ============================================================
# 3. SHOCK WINDOWS
# ============================================================

# COVID recession:
# NBER peak = February 2020
# NBER trough = April 2020
#
# Contraction months used in this study:
# March 2020 - April 2020

COVID_START = 202003
COVID_END = 202004


# Mortgage-rate tightening window:
# Research-defined window based on the sharp increase
# in US mortgage rates from early 2022 through October 2023.

RATE_START = 202201
RATE_END = 202310


# ============================================================
# 4. COMPARISON WINDOWS
# ============================================================

# For descriptive comparison, use windows of the SAME LENGTH
# immediately before and after each shock.
#
# COVID shock length = 2 months
# Before: Jan-Feb 2020
# During: Mar-Apr 2020
# After : May-Jun 2020

SHOCK_WINDOWS = {

    "COVID": {

        "BEFORE": (
            202001,
            202002
        ),

        "DURING": (
            202003,
            202004
        ),

        "AFTER": (
            202005,
            202006
        ),
    },

    # Rate shock length = 22 months
    #
    # Before:
    # Mar 2020 - Dec 2021
    #
    # During:
    # Jan 2022 - Oct 2023
    #
    # After:
    # Nov 2023 - Aug 2025
    #
    # These equal-length windows are a study design choice.
    #
    # IMPORTANT:
    # The "before" rate window overlaps the COVID recession
    # at its beginning. Results are descriptive and should
    # not be interpreted as isolated causal effects.

    "RATE_SHOCK": {

        "BEFORE": (
            202003,
            202112
        ),

        "DURING": (
            202201,
            202310
        ),

        "AFTER": (
            202311,
            202508
        ),
    },
}


# ============================================================
# 5. RAW PERFORMANCE POSITIONS
# ============================================================

LOAN_ID_INDEX = 0
PERIOD_INDEX = 1
DELINQUENCY_INDEX = 3
ZERO_BALANCE_CODE_INDEX = 8


# ============================================================
# 6. DATE FUNCTIONS
# ============================================================

def parse_period(value):

    value = str(value).strip()

    if (
        len(value) != 6
        or not value.isdigit()
    ):
        return None

    year = int(value[:4])
    month = int(value[4:])

    if not (
        1900 <= year <= 2100
        and 1 <= month <= 12
    ):
        return None

    return int(value)


def month_index(period):

    year = period // 100
    month = period % 100

    return (
        year * 12
        + month
    )


def month_difference(
    start_period,
    end_period
):

    if (
        start_period is None
        or end_period is None
    ):
        return None

    return (
        month_index(end_period)
        - month_index(start_period)
    )


def in_period_range(
    period,
    start,
    end
):

    if period is None:
        return False

    return (
        month_index(start)
        <= month_index(period)
        <= month_index(end)
    )


# ============================================================
# 7. LOAD SURVIVAL MASTER
# ============================================================

def load_survival_master():

    loans = {}

    with SURVIVAL_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "loan_id",
            "first_observed_period",
            "event_period",
            "event_code",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                "Missing survival columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            loan_id = (
                row["loan_id"].strip()
            )

            loans[loan_id] = {

                "first_observed_period":
                    parse_period(
                        row[
                            "first_observed_period"
                        ]
                    ),

                "event_period":
                    parse_period(
                        row[
                            "event_period"
                        ]
                    ),

                "event_code":
                    int(
                        float(
                            row[
                                "event_code"
                            ]
                        )
                    ),
            }

    return loans


# ============================================================
# 8. FIND PERFORMANCE MEMBER
# ============================================================

def find_performance_member(
    zip_file
):

    candidates = [

        name

        for name
        in zip_file.namelist()

        if (
            "perf" in name.lower()
            and name.lower().endswith(
                ".txt"
            )
        )
    ]

    if len(candidates) != 1:

        raise ValueError(
            "Expected exactly one "
            "performance TXT file."
        )

    return candidates[0]


# ============================================================
# 9. PARSE DELINQUENCY
# ============================================================

def parse_delinquency(value):

    value = str(value).strip()

    if value.isdigit():
        return int(value)

    # RA / XX remain nonnumeric
    return None


# ============================================================
# 10. IDENTIFY SHOCK PERIOD
# ============================================================

def classify_period(
    period,
    shock_name
):

    windows = (
        SHOCK_WINDOWS[
            shock_name
        ]
    )

    for phase in [
        "BEFORE",
        "DURING",
        "AFTER",
    ]:

        start, end = (
            windows[
                phase
            ]
        )

        if in_period_range(
            period,
            start,
            end
        ):

            return phase

    return None


# ============================================================
# 11. AGGREGATION STRUCTURE
# ============================================================

def new_bucket():

    return {

        "loan_months":
            0,

        "numeric_delinquency_months":
            0,

        "dpd30_months":
            0,

        "dpd60_months":
            0,

        "dpd90_months":
            0,

        "default_events":
            0,

        "prepayment_events":
            0,

        "unique_loans":
            set(),
    }


# ============================================================
# 12. PROCESS PERFORMANCE
# ============================================================

def process_performance(
    loans
):

    results = {

        shock: {

            phase:
                new_bucket()

            for phase in [
                "BEFORE",
                "DURING",
                "AFTER",
            ]
        }

        for shock
        in SHOCK_WINDOWS
    }

    zip_paths = sorted(
        RAW_DIR.glob(
            "sample_*.zip"
        )
    )

    if not zip_paths:

        raise FileNotFoundError(
            f"No raw ZIP files found in "
            f"{RAW_DIR}"
        )

    total_rows = 0

    eligible_first36_rows = 0

    invalid_periods = 0

    for zip_path in zip_paths:

        print(
            f"Processing "
            f"{zip_path.name}..."
        )

        vintage_rows = 0

        vintage_first36 = 0

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zf:

            member = (
                find_performance_member(
                    zf
                )
            )

            with zf.open(
                member,
                "r"
            ) as raw_file:

                for raw_line in raw_file:

                    total_rows += 1
                    vintage_rows += 1

                    line = (
                        raw_line.decode(
                            "utf-8",
                            errors="replace"
                        )
                        .rstrip(
                            "\r\n"
                        )
                    )

                    fields = (
                        line.split("|")
                    )

                    if len(fields) <= max(
                        LOAN_ID_INDEX,
                        PERIOD_INDEX,
                        DELINQUENCY_INDEX,
                        ZERO_BALANCE_CODE_INDEX,
                    ):
                        continue

                    loan_id = (
                        fields[
                            LOAN_ID_INDEX
                        ].strip()
                    )

                    loan = loans.get(
                        loan_id
                    )

                    if loan is None:
                        continue

                    period = parse_period(
                        fields[
                            PERIOD_INDEX
                        ]
                    )

                    if period is None:

                        invalid_periods += 1
                        continue

                    first_period = (
                        loan[
                            "first_observed_period"
                        ]
                    )

                    elapsed = (
                        month_difference(
                            first_period,
                            period
                        )
                    )

                    # Only evaluate behavior
                    # within each loan's
                    # first 36 observed months.

                    if (
                        elapsed is None
                        or elapsed < 0
                        or elapsed >= FIRST_36_MONTHS
                    ):
                        continue

                    eligible_first36_rows += 1
                    vintage_first36 += 1

                    delinquency = (
                        parse_delinquency(
                            fields[
                                DELINQUENCY_INDEX
                            ]
                        )
                    )

                    event_period = (
                        loan[
                            "event_period"
                        ]
                    )

                    event_code = (
                        loan[
                            "event_code"
                        ]
                    )

                    # ----------------------------------------
                    # Evaluate both selected shocks
                    # ----------------------------------------

                    for shock_name in (
                        SHOCK_WINDOWS
                    ):

                        phase = (
                            classify_period(
                                period,
                                shock_name
                            )
                        )

                        if phase is None:
                            continue

                        bucket = (
                            results[
                                shock_name
                            ][
                                phase
                            ]
                        )

                        bucket[
                            "loan_months"
                        ] += 1

                        bucket[
                            "unique_loans"
                        ].add(
                            loan_id
                        )

                        # ------------------------------------
                        # Delinquency prevalence
                        # ------------------------------------

                        if (
                            delinquency
                            is not None
                        ):

                            bucket[
                                "numeric_delinquency_months"
                            ] += 1

                            if (
                                delinquency >= 1
                            ):

                                bucket[
                                    "dpd30_months"
                                ] += 1

                            if (
                                delinquency >= 2
                            ):

                                bucket[
                                    "dpd60_months"
                                ] += 1

                            if (
                                delinquency >= 3
                            ):

                                bucket[
                                    "dpd90_months"
                                ] += 1

                        # ------------------------------------
                        # First Default event
                        # ------------------------------------

                        if (
                            event_code
                            == DEFAULT_EVENT
                            and event_period
                            == period
                        ):

                            bucket[
                                "default_events"
                            ] += 1

                        # ------------------------------------
                        # First Prepayment event
                        # ------------------------------------

                        if (
                            event_code
                            == PREPAY_EVENT
                            and event_period
                            == period
                        ):

                            bucket[
                                "prepayment_events"
                            ] += 1

        print(
            f"  Raw rows      : "
            f"{vintage_rows:,}"
        )

        print(
            f"  First-36 rows : "
            f"{vintage_first36:,}"
        )

    print(
        "\nRaw scan complete."
    )

    print(
        f"Total raw rows       : "
        f"{total_rows:,}"
    )

    print(
        f"First-36 rows        : "
        f"{eligible_first36_rows:,}"
    )

    print(
        f"Invalid PERIOD values: "
        f"{invalid_periods:,}"
    )

    return results


# ============================================================
# 13. BUILD OUTPUT
# ============================================================

def safe_rate(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def build_output(
    results
):

    rows = []

    for shock_name in [
        "COVID",
        "RATE_SHOCK",
    ]:

        for phase in [
            "BEFORE",
            "DURING",
            "AFTER",
        ]:

            bucket = (
                results[
                    shock_name
                ][
                    phase
                ]
            )

            start, end = (
                SHOCK_WINDOWS[
                    shock_name
                ][
                    phase
                ]
            )

            loan_months = (
                bucket[
                    "loan_months"
                ]
            )

            numeric_months = (
                bucket[
                    "numeric_delinquency_months"
                ]
            )

            row = {

                "shock":
                    shock_name,

                "phase":
                    phase,

                "start_period":
                    start,

                "end_period":
                    end,

                "unique_loans":
                    len(
                        bucket[
                            "unique_loans"
                        ]
                    ),

                "loan_months":
                    loan_months,

                "numeric_delinquency_months":
                    numeric_months,

                "dpd30_months":
                    bucket[
                        "dpd30_months"
                    ],

                "dpd30_rate":
                    safe_rate(
                        bucket[
                            "dpd30_months"
                        ],
                        numeric_months
                    ),

                "dpd60_months":
                    bucket[
                        "dpd60_months"
                    ],

                "dpd60_rate":
                    safe_rate(
                        bucket[
                            "dpd60_months"
                        ],
                        numeric_months
                    ),

                "dpd90_months":
                    bucket[
                        "dpd90_months"
                    ],

                "dpd90_rate":
                    safe_rate(
                        bucket[
                            "dpd90_months"
                        ],
                        numeric_months
                    ),

                "default_events":
                    bucket[
                        "default_events"
                    ],

                # Descriptive event count
                # per observed loan-month.
                "default_event_rate":
                    safe_rate(
                        bucket[
                            "default_events"
                        ],
                        loan_months
                    ),

                "prepayment_events":
                    bucket[
                        "prepayment_events"
                    ],

                "prepayment_event_rate":
                    safe_rate(
                        bucket[
                            "prepayment_events"
                        ],
                        loan_months
                    ),
            }

            rows.append(
                row
            )

    return rows


# ============================================================
# 14. VALIDATION
# ============================================================

def validate_output(
    rows
):

    errors = []

    for row in rows:

        for field in [
            "dpd30_rate",
            "dpd60_rate",
            "dpd90_rate",
            "default_event_rate",
            "prepayment_event_rate",
        ]:

            value = row[
                field
            ]

            if not (
                0 <= value <= 1
            ):

                errors.append(
                    (
                        row["shock"],
                        row["phase"],
                        field,
                        value,
                    )
                )

        # Nested delinquency thresholds
        # must satisfy:
        # 30+ >= 60+ >= 90+

        if not (
            row[
                "dpd30_months"
            ]
            >= row[
                "dpd60_months"
            ]
            >= row[
                "dpd90_months"
            ]
        ):

            errors.append(
                (
                    row["shock"],
                    row["phase"],
                    "DPD_THRESHOLD_ORDER",
                    None,
                )
            )

    return errors


# ============================================================
# 15. SAVE OUTPUT
# ============================================================

def save_output(
    rows
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 16. PRINT SUMMARY
# ============================================================

def print_summary(
    rows
):

    print(
        "\n"
        + "=" * 112
    )

    print(
        "BEFORE / DURING / AFTER "
        "SHOCK ANALYSIS"
    )

    print(
        "=" * 112
    )

    print(
        f"{'Shock':<14}"
        f"{'Phase':<10}"
        f"{'Loans':>10}"
        f"{'Loan-Months':>14}"
        f"{'30+ DPD':>12}"
        f"{'60+ DPD':>12}"
        f"{'90+ DPD':>12}"
        f"{'Default':>12}"
        f"{'Prepay':>12}"
    )

    print(
        "-" * 112
    )

    for row in rows:

        print(
            f"{row['shock']:<14}"
            f"{row['phase']:<10}"
            f"{row['unique_loans']:>10,}"
            f"{row['loan_months']:>14,}"
            f"{row['dpd30_rate'] * 100:>11.3f}%"
            f"{row['dpd60_rate'] * 100:>11.3f}%"
            f"{row['dpd90_rate'] * 100:>11.3f}%"
            f"{row['default_event_rate'] * 100:>11.3f}%"
            f"{row['prepayment_event_rate'] * 100:>11.3f}%"
        )

    print(
        "\nRates are calculated per observed "
        "loan-month."
    )

    print(
        "Default and prepayment rates shown here "
        "are descriptive event frequencies, "
        "not cumulative incidence probabilities."
    )

    print(
        "Therefore they must NOT be interpreted "
        "as PD12, PD24, or PD36."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Before/during/after differences are "
        "descriptive associations."
    )

    print(
        "Portfolio composition, vintage, seasoning, "
        "borrower characteristics, censoring, "
        "and other macro conditions may differ "
        "between periods."
    )

    print(
        "No causal effect is inferred."
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK"
    )

    print(
        "STEP 12 - SHOCK PERIOD ANALYSIS\n"
    )

    print(
        "Analysis restricted to each loan's "
        "first 36 observed months."
    )

    print(
        "\nCOVID windows:"
    )

    print(
        "  BEFORE : 202001-202002"
    )

    print(
        "  DURING : 202003-202004"
    )

    print(
        "  AFTER  : 202005-202006"
    )

    print(
        "\nRate-shock windows:"
    )

    print(
        "  BEFORE : 202003-202112"
    )

    print(
        "  DURING : 202201-202310"
    )

    print(
        "  AFTER  : 202311-202508"
    )

    if not SURVIVAL_FILE.exists():

        raise FileNotFoundError(
            f"Missing survival dataset:\n"
            f"{SURVIVAL_FILE}"
        )

    loans = (
        load_survival_master()
    )

    print(
        f"\nSurvival loans loaded: "
        f"{len(loans):,}"
    )

    results = (
        process_performance(
            loans
        )
    )

    rows = (
        build_output(
            results
        )
    )

    errors = (
        validate_output(
            rows
        )
    )

    print_summary(
        rows
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Validation errors: "
        f"{len(errors):,}"
    )

    if errors:

        for error in errors[:20]:

            print(
                "  ",
                error
            )

        raise ValueError(
            "Shock-period validation failed."
        )

    save_output(
        rows
    )

    print(
        f"\nOutput saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nShock period analysis complete."
    )


if __name__ == "__main__":
    main()