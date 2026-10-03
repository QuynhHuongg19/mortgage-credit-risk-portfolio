from pathlib import Path
import csv
import zipfile
from collections import defaultdict


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

SURVIVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

OUTPUT_LOAN = (
    PROJECT_ROOT
    / "outputs"
    / "loan_shock_exposure.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "shock_exposure_summary.csv"
)


# ============================================================
# 2. SHOCK DEFINITIONS
# ============================================================

# Official NBER recession window:
# Peak = February 2020
# Trough = April 2020
# NBER convention:
# recession months = March 2020 through April 2020.

COVID_START = 202003
COVID_END = 202004


# Research-defined mortgage-rate tightening window.
#
# Federal Reserve documentation reports that
# 30-year fixed mortgage rates increased from
# approximately 3.2% in January 2022 to almost
# 8% in October 2023.
#
# This is NOT classified here as an NBER recession.
# It is used as a mortgage-market stress window.

RATE_SHOCK_START = 202201
RATE_SHOCK_END = 202310


# First 36 months of observed loan performance.
#
# With month 0 as the first observed performance month,
# the first 36 observed months correspond to
# elapsed months 0 through 35.

EXPOSURE_WINDOW_MONTHS = 36


# ============================================================
# 3. PERFORMANCE FILE POSITIONS
# ============================================================

# Freddie Mac performance file:
#
# 0 = LOAN IDENTIFIER
# 1 = PERIOD
# 3 = CURRENT LOAN DELINQUENCY STATUS
#
# These positions are consistent with the official
# performance header already validated in Step 01.

LOAN_ID_INDEX = 0
PERIOD_INDEX = 1
DELINQUENCY_INDEX = 3


# ============================================================
# 4. DATE UTILITIES
# ============================================================

def parse_period(value):
    """
    Parse YYYYMM into integer.
    """

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


def month_difference(
    start_period,
    end_period
):
    """
    Calendar-month difference.

    Example:
    202001 -> 202001 = 0
    202001 -> 202002 = 1
    202001 -> 202101 = 12
    """

    if (
        start_period is None
        or end_period is None
    ):
        return None

    start_year = start_period // 100
    start_month = start_period % 100

    end_year = end_period // 100
    end_month = end_period % 100

    return (
        (end_year - start_year) * 12
        + (end_month - start_month)
    )


def period_in_window(
    period,
    start_period,
    end_period
):
    """
    True if period falls inside an inclusive
    calendar shock window.
    """

    return (
        period is not None
        and start_period <= period <= end_period
    )


# ============================================================
# 5. DELINQUENCY UTILITY
# ============================================================

def parse_delinquency(value):
    """
    Freddie Mac numeric delinquency values represent
    number of payments past due.

    RA and XX remain nonnumeric and are not coerced.
    """

    value = str(value).strip()

    if value.isdigit():
        return int(value)

    return None


# ============================================================
# 6. LOAD SURVIVAL MASTER
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
            "vintage",
            "first_observed_period",
            "event_period",
            "duration_months",
            "event_code",
            "event_type",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                "Survival dataset missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            loan_id = (
                row["loan_id"]
                .strip()
            )

            first_period = parse_period(
                row["first_observed_period"]
            )

            event_period = parse_period(
                row["event_period"]
            )

            duration = int(
                float(
                    row["duration_months"]
                )
            )

            event_code = int(
                float(
                    row["event_code"]
                )
            )

            loans[loan_id] = {

                "loan_id":
                    loan_id,

                "vintage":
                    row["vintage"].strip(),

                "first_observed_period":
                    first_period,

                "event_period":
                    event_period,

                "duration_months":
                    duration,

                "event_code":
                    event_code,

                "event_type":
                    row["event_type"].strip(),

                # Exposure counters
                "observed_months_first36":
                    0,

                "covid_exposed":
                    0,

                "covid_months_exposed":
                    0,

                "rate_shock_exposed":
                    0,

                "rate_shock_months_exposed":
                    0,

                # Delinquency behavior
                "ever_30dpd_first36":
                    0,

                "ever_60dpd_first36":
                    0,

                "ever_90dpd_first36":
                    0,

                "ever_30dpd_during_covid":
                    0,

                "ever_60dpd_during_covid":
                    0,

                "ever_90dpd_during_covid":
                    0,

                "ever_30dpd_during_rate_shock":
                    0,

                "ever_60dpd_during_rate_shock":
                    0,

                "ever_90dpd_during_rate_shock":
                    0,
            }

    return loans


# ============================================================
# 7. FIND PERFORMANCE FILE
# ============================================================

def find_performance_member(
    zip_file
):
    """
    Return the performance TXT member
    from one sample vintage ZIP.
    """

    members = zip_file.namelist()

    candidates = [
        name
        for name in members
        if (
            "perf" in name.lower()
            and name.lower().endswith(".txt")
        )
    ]

    if len(candidates) != 1:

        raise ValueError(
            "Expected exactly one performance "
            f"file, found {len(candidates)}."
        )

    return candidates[0]


# ============================================================
# 8. PROCESS RAW PERFORMANCE
# ============================================================

def process_performance(
    loans
):

    zip_paths = sorted(
        RAW_DIR.glob(
            "sample_*.zip"
        )
    )

    if not zip_paths:

        raise FileNotFoundError(
            f"No sample ZIP files found in:\n"
            f"{RAW_DIR}"
        )

    total_rows = 0
    matched_rows = 0
    first36_rows = 0
    invalid_periods = 0

    for zip_path in zip_paths:

        print(
            f"Processing {zip_path.name}..."
        )

        vintage_rows = 0
        vintage_first36 = 0

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zf:

            perf_member = (
                find_performance_member(
                    zf
                )
            )

            with zf.open(
                perf_member,
                "r"
            ) as raw_file:

                for raw_line in raw_file:

                    total_rows += 1
                    vintage_rows += 1

                    line = raw_line.decode(
                        "utf-8",
                        errors="replace"
                    ).rstrip(
                        "\r\n"
                    )

                    fields = line.split("|")

                    if len(fields) <= max(
                        LOAN_ID_INDEX,
                        PERIOD_INDEX,
                        DELINQUENCY_INDEX
                    ):
                        continue

                    loan_id = (
                        fields[
                            LOAN_ID_INDEX
                        ]
                        .strip()
                    )

                    loan = loans.get(
                        loan_id
                    )

                    if loan is None:
                        continue

                    matched_rows += 1

                    period = parse_period(
                        fields[
                            PERIOD_INDEX
                        ]
                    )

                    if period is None:

                        invalid_periods += 1
                        continue

                    first_period = loan[
                        "first_observed_period"
                    ]

                    elapsed = month_difference(
                        first_period,
                        period
                    )

                    if elapsed is None:
                        continue

                    # Only first 36 observed calendar months:
                    # elapsed month 0 through 35.

                    if not (
                        0 <= elapsed
                        < EXPOSURE_WINDOW_MONTHS
                    ):
                        continue

                    first36_rows += 1
                    vintage_first36 += 1

                    loan[
                        "observed_months_first36"
                    ] += 1

                    delinquency = (
                        parse_delinquency(
                            fields[
                                DELINQUENCY_INDEX
                            ]
                        )
                    )

                    # ----------------------------------------
                    # Overall first-36-month delinquency
                    # ----------------------------------------

                    if delinquency is not None:

                        if delinquency >= 1:

                            loan[
                                "ever_30dpd_first36"
                            ] = 1

                        if delinquency >= 2:

                            loan[
                                "ever_60dpd_first36"
                            ] = 1

                        if delinquency >= 3:

                            loan[
                                "ever_90dpd_first36"
                            ] = 1

                    # ----------------------------------------
                    # COVID recession exposure
                    # ----------------------------------------

                    if period_in_window(
                        period,
                        COVID_START,
                        COVID_END
                    ):

                        loan[
                            "covid_exposed"
                        ] = 1

                        loan[
                            "covid_months_exposed"
                        ] += 1

                        if delinquency is not None:

                            if delinquency >= 1:

                                loan[
                                    "ever_30dpd_during_covid"
                                ] = 1

                            if delinquency >= 2:

                                loan[
                                    "ever_60dpd_during_covid"
                                ] = 1

                            if delinquency >= 3:

                                loan[
                                    "ever_90dpd_during_covid"
                                ] = 1

                    # ----------------------------------------
                    # Mortgage-rate shock exposure
                    # ----------------------------------------

                    if period_in_window(
                        period,
                        RATE_SHOCK_START,
                        RATE_SHOCK_END
                    ):

                        loan[
                            "rate_shock_exposed"
                        ] = 1

                        loan[
                            "rate_shock_months_exposed"
                        ] += 1

                        if delinquency is not None:

                            if delinquency >= 1:

                                loan[
                                    "ever_30dpd_during_rate_shock"
                                ] = 1

                            if delinquency >= 2:

                                loan[
                                    "ever_60dpd_during_rate_shock"
                                ] = 1

                            if delinquency >= 3:

                                loan[
                                    "ever_90dpd_during_rate_shock"
                                ] = 1

        print(
            f"  Rows             : "
            f"{vintage_rows:,}"
        )

        print(
            f"  First-36 rows    : "
            f"{vintage_first36:,}"
        )

    print(
        "\nRaw performance scan complete."
    )

    print(
        f"Performance rows     : "
        f"{total_rows:,}"
    )

    print(
        f"Matched survival rows: "
        f"{matched_rows:,}"
    )

    print(
        f"First-36 rows        : "
        f"{first36_rows:,}"
    )

    print(
        f"Invalid periods      : "
        f"{invalid_periods:,}"
    )


# ============================================================
# 9. DERIVE LOAN-LEVEL SHOCK FEATURES
# ============================================================

def derive_loan_features(
    loans
):

    rows = []

    for loan in loans.values():

        duration = loan[
            "duration_months"
        ]

        event_code = loan[
            "event_code"
        ]

        event_period = loan[
            "event_period"
        ]

        # ----------------------------------------
        # Study event within first 36 months
        # ----------------------------------------

        default_within_36 = int(
            event_code == DEFAULT_EVENT
            and duration <= 35
        )

        prepay_within_36 = int(
            event_code == PREPAY_EVENT
            and duration <= 35
        )

        # ----------------------------------------
        # Did Default occur during a shock window?
        # ----------------------------------------

        default_during_covid = int(
            event_code == DEFAULT_EVENT
            and duration <= 35
            and period_in_window(
                event_period,
                COVID_START,
                COVID_END
            )
        )

        default_during_rate_shock = int(
            event_code == DEFAULT_EVENT
            and duration <= 35
            and period_in_window(
                event_period,
                RATE_SHOCK_START,
                RATE_SHOCK_END
            )
        )

        # ----------------------------------------
        # Did Prepayment occur during shock?
        # ----------------------------------------

        prepay_during_covid = int(
            event_code == PREPAY_EVENT
            and duration <= 35
            and period_in_window(
                event_period,
                COVID_START,
                COVID_END
            )
        )

        prepay_during_rate_shock = int(
            event_code == PREPAY_EVENT
            and duration <= 35
            and period_in_window(
                event_period,
                RATE_SHOCK_START,
                RATE_SHOCK_END
            )
        )

        # ----------------------------------------
        # Exposure group
        # ----------------------------------------

        covid = loan[
            "covid_exposed"
        ]

        rate = loan[
            "rate_shock_exposed"
        ]

        if covid and rate:

            shock_group = (
                "COVID_AND_RATE_SHOCK"
            )

        elif covid:

            shock_group = (
                "COVID_ONLY"
            )

        elif rate:

            shock_group = (
                "RATE_SHOCK_ONLY"
            )

        else:

            shock_group = (
                "NO_SELECTED_SHOCK"
            )

        row = dict(loan)

        row.update(
            {
                "shock_group":
                    shock_group,

                "default_within_36":
                    default_within_36,

                "prepay_within_36":
                    prepay_within_36,

                "default_during_covid":
                    default_during_covid,

                "prepay_during_covid":
                    prepay_during_covid,

                "default_during_rate_shock":
                    default_during_rate_shock,

                "prepay_during_rate_shock":
                    prepay_during_rate_shock,
            }
        )

        rows.append(row)

    return rows


# ============================================================
# 10. BUILD SUMMARY
# ============================================================

def build_summary(
    rows
):

    summary = defaultdict(
        lambda: {
            "loans": 0,
            "default_within_36": 0,
            "prepay_within_36": 0,
            "ever_30dpd_first36": 0,
            "ever_60dpd_first36": 0,
            "ever_90dpd_first36": 0,
        }
    )

    for row in rows:

        group = row[
            "shock_group"
        ]

        summary[
            group
        ][
            "loans"
        ] += 1

        for field in [
            "default_within_36",
            "prepay_within_36",
            "ever_30dpd_first36",
            "ever_60dpd_first36",
            "ever_90dpd_first36",
        ]:

            summary[
                group
            ][
                field
            ] += row[
                field
            ]

    output = []

    for group in sorted(
        summary.keys()
    ):

        item = summary[
            group
        ]

        loans = item[
            "loans"
        ]

        result = {
            "shock_group":
                group,

            "loans":
                loans,
        }

        for field in [
            "default_within_36",
            "prepay_within_36",
            "ever_30dpd_first36",
            "ever_60dpd_first36",
            "ever_90dpd_first36",
        ]:

            count = item[
                field
            ]

            pct = (
                count
                / loans
                * 100
                if loans > 0
                else 0.0
            )

            result[
                field
            ] = count

            result[
                f"{field}_pct"
            ] = pct

        output.append(
            result
        )

    return output


# ============================================================
# 11. SAVE CSV
# ============================================================

def save_csv(
    path,
    rows
):

    if not rows:
        return

    with path.open(
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
# 12. MAIN
# ============================================================

DEFAULT_EVENT = 1
PREPAY_EVENT = 2


def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "36-MONTH MACRO SHOCK EXPOSURE\n"
    )

    print(
        "Shock definitions:"
    )

    print(
        "  COVID recession        : "
        "202003-202004"
    )

    print(
        "  Mortgage-rate tightening: "
        "202201-202310"
    )

    print(
        "\nExposure window:"
    )

    print(
        "  First 36 observed calendar months "
        "(elapsed months 0-35)"
    )

    if not SURVIVAL_FILE.exists():

        raise FileNotFoundError(
            f"Survival file not found:\n"
            f"{SURVIVAL_FILE}"
        )

    OUTPUT_LOAN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load master
    # --------------------------------------------------------

    loans = load_survival_master()

    print(
        f"\nSurvival loans loaded: "
        f"{len(loans):,}"
    )

    # --------------------------------------------------------
    # Scan raw performance
    # --------------------------------------------------------

    process_performance(
        loans
    )

    # --------------------------------------------------------
    # Derive features
    # --------------------------------------------------------

    loan_rows = (
        derive_loan_features(
            loans
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    no_observation = sum(
        row[
            "observed_months_first36"
        ] == 0
        for row in loan_rows
    )

    covid_exposed = sum(
        row[
            "covid_exposed"
        ]
        for row in loan_rows
    )

    rate_exposed = sum(
        row[
            "rate_shock_exposed"
        ]
        for row in loan_rows
    )

    both_exposed = sum(
        row[
            "covid_exposed"
        ]
        and row[
            "rate_shock_exposed"
        ]
        for row in loan_rows
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "LOAN-LEVEL EXPOSURE AUDIT"
    )

    print(
        "=" * 78
    )

    print(
        f"Loans                    : "
        f"{len(loan_rows):,}"
    )

    print(
        f"No first-36 observation  : "
        f"{no_observation:,}"
    )

    print(
        f"COVID exposed            : "
        f"{covid_exposed:,}"
    )

    print(
        f"Rate-shock exposed       : "
        f"{rate_exposed:,}"
    )

    print(
        f"Exposed to both          : "
        f"{both_exposed:,}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = (
        build_summary(
            loan_rows
        )
    )

    print(
        "\n"
        + "=" * 108
    )

    print(
        "FIRST-36-MONTH OUTCOME SUMMARY BY SHOCK EXPOSURE"
    )

    print(
        "=" * 108
    )

    print(
        f"{'Shock group':<26}"
        f"{'Loans':>10}"
        f"{'Default36':>14}"
        f"{'Prepay36':>14}"
        f"{'30+ DPD':>14}"
        f"{'60+ DPD':>14}"
        f"{'90+ DPD':>14}"
    )

    print(
        "-" * 108
    )

    for row in summary_rows:

        print(
            f"{row['shock_group']:<26}"
            f"{row['loans']:>10,}"
            f"{row['default_within_36_pct']:>13.2f}%"
            f"{row['prepay_within_36_pct']:>13.2f}%"
            f"{row['ever_30dpd_first36_pct']:>13.2f}%"
            f"{row['ever_60dpd_first36_pct']:>13.2f}%"
            f"{row['ever_90dpd_first36_pct']:>13.2f}%"
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_csv(
        OUTPUT_LOAN,
        loan_rows
    )

    save_csv(
        OUTPUT_SUMMARY,
        summary_rows
    )

    print(
        f"\nLoan-level shock features saved to:\n"
        f"{OUTPUT_LOAN}"
    )

    print(
        f"\nShock summary saved to:\n"
        f"{OUTPUT_SUMMARY}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Shock-group percentages are descriptive."
    )

    print(
        "They are NOT causal estimates of the effect "
        "of macroeconomic shocks on Default."
    )

    print(
        "\nMacroeconomic shock exposure "
        "analysis complete."
    )


if __name__ == "__main__":
    main()