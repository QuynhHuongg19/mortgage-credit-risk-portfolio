from pathlib import Path
from zipfile import ZipFile
import csv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_DIR = PROJECT_ROOT / "docs" / "references"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HEADER_ZIP = (
    REFERENCE_DIR
    / "file_headers_july_2026.zip"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "survival_loan_level.csv"
)


# ============================================================
# 2. EVENT CODING
# ============================================================

# event_code:
# 0 = Censor
# 1 = Default
# 2 = Voluntary Prepayment
#
# Default:
# First observed numeric delinquency status >= 3.
#
# Prepayment:
# ZERO BALANCE CODE = 01.
#
# Same-month Default + Prepayment:
# Default wins under this project's documented tie rule.
#
# Other termination codes:
# 15 / 16 / 96 are treated as censoring if they occur
# before Default or Prepayment.

OTHER_CENSOR_CODES = {
    "15",
    "16",
    "96",
}


# ============================================================
# 3. READ OFFICIAL HEADERS
# ============================================================

def read_headers():

    with ZipFile(
        HEADER_ZIP,
        "r"
    ) as zip_file:

        names = zip_file.namelist()

        orig_header_name = next(
            name
            for name in names
            if "orig" in name.lower()
        )

        perf_header_name = next(
            name
            for name in names
            if "perf" in name.lower()
        )

        with zip_file.open(
            orig_header_name
        ) as file:

            orig_line = (
                file.readline()
                .decode(
                    "utf-8-sig",
                    errors="replace"
                )
                .rstrip("\r\n")
            )

        with zip_file.open(
            perf_header_name
        ) as file:

            perf_line = (
                file.readline()
                .decode(
                    "utf-8-sig",
                    errors="replace"
                )
                .rstrip("\r\n")
            )

    return (
        orig_line.split("|"),
        perf_line.split("|"),
    )


# ============================================================
# 4. COLUMN LOOKUP
# ============================================================

def find_index(
    columns,
    column_name
):

    normalized = [
        column.strip().upper()
        for column in columns
    ]

    target = (
        column_name
        .strip()
        .upper()
    )

    if target not in normalized:

        raise ValueError(
            f"Column not found: "
            f"{column_name}"
        )

    return normalized.index(
        target
    )


# ============================================================
# 5. SAFE CONVERSION
# ============================================================

def clean_value(value):

    value = value.strip()

    if value == "":
        return None

    return value


def to_int(value):

    value = value.strip()

    if value == "":
        return None

    try:
        return int(value)

    except ValueError:
        return None


def to_float(value):

    value = value.strip()

    if value == "":
        return None

    try:
        return float(value)

    except ValueError:
        return None


def period_to_int(value):

    value = value.strip()

    if (
        len(value) == 6
        and value.isdigit()
    ):

        year = int(
            value[:4]
        )

        month = int(
            value[4:]
        )

        if (
            1900 <= year <= 2100
            and 1 <= month <= 12
        ):
            return int(value)

    return None


# ============================================================
# 6. CALENDAR-MONTH TIME SCALE
# ============================================================

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

    if not (
        1 <= start_month <= 12
        and 1 <= end_month <= 12
    ):
        return None

    return (
        (end_year - start_year) * 12
        + (end_month - start_month)
    )


# ============================================================
# 7. READ ORIGINATION DATA
# ============================================================

def read_origination(
    zip_file,
    vintage,
    orig_indexes
):

    file_name = (
        f"sample_orig_{vintage}.txt"
    )

    loans = {}

    with zip_file.open(
        file_name
    ) as file:

        for raw_line in file:

            values = (
                raw_line
                .decode(
                    "utf-8",
                    errors="replace"
                )
                .rstrip("\r\n")
                .split("|")
            )

            loan_id = values[
                orig_indexes[
                    "loan_id"
                ]
            ].strip()

            if not loan_id:
                continue

            loans[loan_id] = {

                "loan_id":
                    loan_id,

                "vintage":
                    int(vintage),

                "fico":
                    to_float(
                        values[
                            orig_indexes[
                                "fico"
                            ]
                        ]
                    ),

                "original_ltv":
                    to_float(
                        values[
                            orig_indexes[
                                "ltv"
                            ]
                        ]
                    ),

                "original_cltv":
                    to_float(
                        values[
                            orig_indexes[
                                "cltv"
                            ]
                        ]
                    ),

                "original_dti":
                    to_float(
                        values[
                            orig_indexes[
                                "dti"
                            ]
                        ]
                    ),

                "original_upb":
                    to_float(
                        values[
                            orig_indexes[
                                "upb"
                            ]
                        ]
                    ),

                "original_interest_rate":
                    to_float(
                        values[
                            orig_indexes[
                                "interest_rate"
                            ]
                        ]
                    ),

                "original_loan_term":
                    to_int(
                        values[
                            orig_indexes[
                                "loan_term"
                            ]
                        ]
                    ),

                "first_payment_date":
                    clean_value(
                        values[
                            orig_indexes[
                                "first_payment_date"
                            ]
                        ]
                    ),
            }

    return loans


# ============================================================
# 8. READ PERFORMANCE HISTORY
# ============================================================

def read_performance(
    zip_file,
    vintage,
    perf_indexes
):

    file_name = (
        f"sample_perf_{vintage}.txt"
    )

    history = {}

    with zip_file.open(
        file_name
    ) as file:

        for raw_line in file:

            values = (
                raw_line
                .decode(
                    "utf-8",
                    errors="replace"
                )
                .rstrip("\r\n")
                .split("|")
            )

            loan_id = values[
                perf_indexes[
                    "loan_id"
                ]
            ].strip()

            if not loan_id:
                continue

            period = period_to_int(
                values[
                    perf_indexes[
                        "period"
                    ]
                ]
            )

            loan_age = to_int(
                values[
                    perf_indexes[
                        "loan_age"
                    ]
                ]
            )

            delinquency_raw = (
                values[
                    perf_indexes[
                        "delinquency"
                    ]
                ]
                .strip()
            )

            zero_balance_code = (
                values[
                    perf_indexes[
                        "zero_balance"
                    ]
                ]
                .strip()
            )

            if loan_id not in history:

                history[loan_id] = {

                    "first_period":
                        None,

                    "last_period":
                        None,

                    "first_loan_age":
                        None,

                    "last_loan_age":
                        None,

                    "default_period":
                        None,

                    "default_loan_age":
                        None,

                    "prepay_period":
                        None,

                    "prepay_loan_age":
                        None,

                    "other_censor_period":
                        None,

                    "other_censor_loan_age":
                        None,

                    "other_censor_code":
                        None,
                }

            record = history[
                loan_id
            ]

            # ------------------------------------------------
            # Observation window
            # ------------------------------------------------

            if period is not None:

                if (
                    record[
                        "first_period"
                    ] is None
                    or period
                    < record[
                        "first_period"
                    ]
                ):

                    record[
                        "first_period"
                    ] = period

                    record[
                        "first_loan_age"
                    ] = loan_age

                if (
                    record[
                        "last_period"
                    ] is None
                    or period
                    > record[
                        "last_period"
                    ]
                ):

                    record[
                        "last_period"
                    ] = period

                    record[
                        "last_loan_age"
                    ] = loan_age

            # ------------------------------------------------
            # Default
            # ------------------------------------------------
            #
            # Numeric delinquency >= 3 corresponds to the
            # project's 90+ DPD event definition.
            #
            # Keep the earliest observed qualifying month.

            if (
                delinquency_raw.isdigit()
                and period is not None
            ):

                delinquency = int(
                    delinquency_raw
                )

                if delinquency >= 3:

                    if (
                        record[
                            "default_period"
                        ] is None
                        or period
                        < record[
                            "default_period"
                        ]
                    ):

                        record[
                            "default_period"
                        ] = period

                        record[
                            "default_loan_age"
                        ] = loan_age

            # ------------------------------------------------
            # Voluntary Prepayment candidate
            # ------------------------------------------------

            if (
                zero_balance_code == "01"
                and period is not None
            ):

                if (
                    record[
                        "prepay_period"
                    ] is None
                    or period
                    < record[
                        "prepay_period"
                    ]
                ):

                    record[
                        "prepay_period"
                    ] = period

                    record[
                        "prepay_loan_age"
                    ] = loan_age

            # ------------------------------------------------
            # Other terminal censoring candidate
            # ------------------------------------------------

            if (
                zero_balance_code
                in OTHER_CENSOR_CODES
                and period is not None
            ):

                if (
                    record[
                        "other_censor_period"
                    ] is None
                    or period
                    < record[
                        "other_censor_period"
                    ]
                ):

                    record[
                        "other_censor_period"
                    ] = period

                    record[
                        "other_censor_loan_age"
                    ] = loan_age

                    record[
                        "other_censor_code"
                    ] = zero_balance_code

    return history


# ============================================================
# 9. DETERMINE FIRST STUDY EVENT
# ============================================================

def determine_event(record):

    default_period = record[
        "default_period"
    ]

    prepay_period = record[
        "prepay_period"
    ]

    other_period = record[
        "other_censor_period"
    ]

    last_period = record[
        "last_period"
    ]

    candidates = []

    # --------------------------------------------------------
    # Default
    #
    # Priority 1 means Default wins if Default and Prepayment
    # are observed in the same calendar month.
    # --------------------------------------------------------

    if default_period is not None:

        candidates.append(
            (
                default_period,
                1,
                "DEFAULT",
                1,
                record[
                    "default_loan_age"
                ],
                None,
            )
        )

    # --------------------------------------------------------
    # Voluntary Prepayment
    # --------------------------------------------------------

    if prepay_period is not None:

        candidates.append(
            (
                prepay_period,
                2,
                "PREPAYMENT",
                2,
                record[
                    "prepay_loan_age"
                ],
                "01",
            )
        )

    # --------------------------------------------------------
    # Other termination censor
    # --------------------------------------------------------

    if other_period is not None:

        candidates.append(
            (
                other_period,
                3,
                "CENSOR_OTHER_TERMINATION",
                0,
                record[
                    "other_censor_loan_age"
                ],
                record[
                    "other_censor_code"
                ],
            )
        )

    # --------------------------------------------------------
    # No terminal study event:
    # right censor at last observed performance period.
    # --------------------------------------------------------

    if not candidates:

        return {

            "event_period":
                last_period,

            "event_loan_age":
                record[
                    "last_loan_age"
                ],

            "event_code":
                0,

            "event_type":
                "CENSOR_END_OF_OBSERVATION",

            "termination_code":
                None,
        }

    # Earliest calendar period wins.
    # Priority resolves same-month ties.

    chosen = min(
        candidates,
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return {

        "event_period":
            chosen[0],

        "event_loan_age":
            chosen[4],

        "event_code":
            chosen[3],

        "event_type":
            chosen[2],

        "termination_code":
            chosen[5],
    }


# ============================================================
# 10. BUILD ONE VINTAGE
# ============================================================

def build_vintage(
    zip_path,
    orig_indexes,
    perf_indexes
):

    vintage = (
        zip_path
        .stem
        .split("_")[-1]
    )

    print(
        f"Processing vintage "
        f"{vintage}..."
    )

    with ZipFile(
        zip_path,
        "r"
    ) as zip_file:

        origination = read_origination(
            zip_file,
            vintage,
            orig_indexes
        )

        performance = read_performance(
            zip_file,
            vintage,
            perf_indexes
        )

    rows = []

    missing_performance = 0
    invalid_duration = 0

    for loan_id, baseline in (
        origination.items()
    ):

        if loan_id not in performance:

            missing_performance += 1
            continue

        history = performance[
            loan_id
        ]

        event = determine_event(
            history
        )

        event_age = event[
            "event_loan_age"
        ]

        # ----------------------------------------------------
        # Survival analysis time scale
        # ----------------------------------------------------
        #
        # Duration is elapsed CALENDAR MONTHS from the first
        # observed performance record to the first study event
        # or censoring period.
        #
        # event_loan_age remains in the dataset as an audit
        # field but is NOT used as survival duration because
        # the raw sequence audit found occasional Loan Age
        # decreases/resets.

        start_period = history[
            "first_period"
        ]

        end_period = event[
            "event_period"
        ]

        duration_months = (
            month_difference(
                start_period,
                end_period
            )
        )

        if (
            duration_months is None
            or duration_months < 0
        ):

            invalid_duration += 1
            continue

        row = {

            **baseline,

            "first_observed_period":
                history[
                    "first_period"
                ],

            "first_observed_loan_age":
                history[
                    "first_loan_age"
                ],

            "last_observed_period":
                history[
                    "last_period"
                ],

            "last_observed_loan_age":
                history[
                    "last_loan_age"
                ],

            "default_period":
                history[
                    "default_period"
                ],

            "default_loan_age":
                history[
                    "default_loan_age"
                ],

            "prepay_period":
                history[
                    "prepay_period"
                ],

            "prepay_loan_age":
                history[
                    "prepay_loan_age"
                ],

            "event_period":
                event[
                    "event_period"
                ],

            "event_loan_age":
                event_age,

            "duration_months":
                duration_months,

            "event_code":
                event[
                    "event_code"
                ],

            "event_type":
                event[
                    "event_type"
                ],

            "termination_code":
                event[
                    "termination_code"
                ],
        }

        rows.append(
            row
        )

    print(
        f"  Survival rows      : "
        f"{len(rows):,}"
    )

    print(
        f"  Missing performance: "
        f"{missing_performance:,}"
    )

    print(
        f"  Invalid duration   : "
        f"{invalid_duration:,}"
    )

    return (
        rows,
        missing_performance,
        invalid_duration,
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "BUILD SURVIVAL DATASET\n"
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    orig_columns, perf_columns = (
        read_headers()
    )

    # --------------------------------------------------------
    # Origination indexes
    # --------------------------------------------------------

    orig_indexes = {

        "loan_id":
            find_index(
                orig_columns,
                "LOAN IDENTIFIER"
            ),

        "fico":
            find_index(
                orig_columns,
                "CLASSIC FICO"
            ),

        "first_payment_date":
            find_index(
                orig_columns,
                "FIRST PAYMENT DATE"
            ),

        "cltv":
            find_index(
                orig_columns,
                "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)"
            ),

        "dti":
            find_index(
                orig_columns,
                "ORIGINAL DEBT-TO-INCOME (DTI) RATIO"
            ),

        "upb":
            find_index(
                orig_columns,
                "ORIGINAL UPB"
            ),

        "ltv":
            find_index(
                orig_columns,
                "ORIGINAL LOAN-TO-VALUE (LTV)"
            ),

        "interest_rate":
            find_index(
                orig_columns,
                "ORIGINAL INTEREST RATE"
            ),

        "loan_term":
            find_index(
                orig_columns,
                "ORIGINAL LOAN TERM"
            ),
    }

    # --------------------------------------------------------
    # Performance indexes
    # --------------------------------------------------------

    perf_indexes = {

        "loan_id":
            find_index(
                perf_columns,
                "LOAN IDENTIFIER"
            ),

        "period":
            find_index(
                perf_columns,
                "PERIOD"
            ),

        "delinquency":
            find_index(
                perf_columns,
                "CURRENT LOAN DELINQUENCY STATUS"
            ),

        "loan_age":
            find_index(
                perf_columns,
                "LOAN AGE"
            ),

        "zero_balance":
            find_index(
                perf_columns,
                "ZERO BALANCE CODE"
            ),
    }

    # --------------------------------------------------------
    # Raw ZIP files
    # --------------------------------------------------------

    zip_files = sorted(
        RAW_DIR.glob(
            "sample_*.zip"
        )
    )

    if not zip_files:

        raise FileNotFoundError(
            "No sample ZIP files found."
        )

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    fieldnames = [

        "loan_id",
        "vintage",

        "fico",
        "original_ltv",
        "original_cltv",
        "original_dti",
        "original_upb",
        "original_interest_rate",
        "original_loan_term",
        "first_payment_date",

        "first_observed_period",
        "first_observed_loan_age",

        "last_observed_period",
        "last_observed_loan_age",

        "default_period",
        "default_loan_age",

        "prepay_period",
        "prepay_loan_age",

        "event_period",
        "event_loan_age",

        "duration_months",

        "event_code",
        "event_type",
        "termination_code",
    ]

    total_rows = 0
    total_missing = 0
    total_invalid_duration = 0

    event_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    event_type_counts = {}

    # --------------------------------------------------------
    # Build master loan-level dataset
    # --------------------------------------------------------

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

        for zip_path in zip_files:

            (
                rows,
                missing,
                invalid_duration,
            ) = build_vintage(
                zip_path,
                orig_indexes,
                perf_indexes
            )

            writer.writerows(
                rows
            )

            total_rows += len(
                rows
            )

            total_missing += (
                missing
            )

            total_invalid_duration += (
                invalid_duration
            )

            for row in rows:

                event_code = row[
                    "event_code"
                ]

                event_counts[
                    event_code
                ] += 1

                event_type = row[
                    "event_type"
                ]

                event_type_counts[
                    event_type
                ] = (
                    event_type_counts.get(
                        event_type,
                        0
                    )
                    + 1
                )

    # ========================================================
    # 12. FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SURVIVAL DATASET SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Survival cohort      : "
        f"{total_rows:,}"
    )

    print(
        f"Excluded no perf     : "
        f"{total_missing:,}"
    )

    print(
        f"Invalid duration     : "
        f"{total_invalid_duration:,}"
    )

    print(
        f"Default events       : "
        f"{event_counts[1]:,}"
    )

    print(
        f"Prepayment events    : "
        f"{event_counts[2]:,}"
    )

    print(
        f"Censored observations: "
        f"{event_counts[0]:,}"
    )

    print(
        "\nEvent type breakdown:"
    )

    for event_type, count in sorted(
        event_type_counts.items()
    ):

        print(
            f"  {event_type:<30}"
            f"{count:>10,}"
        )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nTime scale:"
    )

    print(
        "  duration_months = elapsed calendar months"
    )

    print(
        "  from first observed performance period"
    )

    print(
        "  to first study event or censoring period."
    )

    print(
        "\nSurvival dataset build complete."
    )


if __name__ == "__main__":
    main()