from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import csv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_DIR = PROJECT_ROOT / "docs" / "references"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

HEADER_ZIP = REFERENCE_DIR / "file_headers_july_2026.zip"


# ============================================================
# 2. READ OFFICIAL PERFORMANCE HEADER
# ============================================================

def read_performance_header():

    if not HEADER_ZIP.exists():
        raise FileNotFoundError(
            f"Header ZIP not found: {HEADER_ZIP}"
        )

    with ZipFile(HEADER_ZIP, "r") as zip_file:

        names = zip_file.namelist()

        perf_header_file = next(
            name for name in names
            if "perf" in name.lower()
        )

        with zip_file.open(perf_header_file) as file:

            header_line = file.readline().decode(
                "utf-8-sig",
                errors="replace"
            ).rstrip("\r\n")

    return header_line.split("|")


# ============================================================
# 3. FIND COLUMN INDEX
# ============================================================

def find_column_index(columns, target_name):

    normalized = [
        column.strip().upper()
        for column in columns
    ]

    target = target_name.strip().upper()

    if target not in normalized:
        raise ValueError(
            f"Column not found: {target_name}"
        )

    return normalized.index(target)


# ============================================================
# 4. PERIOD CONVERSION
# ============================================================

def period_to_int(period):
    """
    Convert YYYYMM to integer for chronological comparison.
    """

    period = period.strip()

    if len(period) != 6 or not period.isdigit():
        return None

    return int(period)


# ============================================================
# 5. AUDIT ONE VINTAGE
# ============================================================

def audit_vintage(
    zip_path,
    loan_id_index,
    period_index,
    delinquency_index,
    zero_balance_index
):

    vintage = zip_path.stem.split("_")[-1]

    performance_file = (
        f"sample_perf_{vintage}.txt"
    )

    first_observed = {}
    last_observed = {}

    first_90 = {}
    prepayment = {}

    with ZipFile(zip_path, "r") as zip_file:

        if performance_file not in zip_file.namelist():
            raise ValueError(
                f"{performance_file} not found."
            )

        with zip_file.open(performance_file) as file:

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
                    loan_id_index
                ].strip()

                period_raw = values[
                    period_index
                ].strip()

                delinquency_raw = values[
                    delinquency_index
                ].strip()

                zero_balance_code = values[
                    zero_balance_index
                ].strip()

                period = period_to_int(
                    period_raw
                )

                if period is None:
                    continue

                # --------------------------------------------
                # Observation window
                # --------------------------------------------

                if loan_id not in first_observed:
                    first_observed[loan_id] = period
                else:
                    first_observed[loan_id] = min(
                        first_observed[loan_id],
                        period
                    )

                if loan_id not in last_observed:
                    last_observed[loan_id] = period
                else:
                    last_observed[loan_id] = max(
                        last_observed[loan_id],
                        period
                    )

                # --------------------------------------------
                # First 90+ DPD
                # Numeric status >= 3
                # --------------------------------------------

                if delinquency_raw.isdigit():

                    delinquency = int(
                        delinquency_raw
                    )

                    if delinquency >= 3:

                        if loan_id not in first_90:
                            first_90[
                                loan_id
                            ] = period

                        else:
                            first_90[
                                loan_id
                            ] = min(
                                first_90[loan_id],
                                period
                            )

                # --------------------------------------------
                # Voluntary prepayment
                # ZERO BALANCE CODE = 01
                # --------------------------------------------

                if zero_balance_code == "01":

                    if loan_id not in prepayment:
                        prepayment[
                            loan_id
                        ] = period

                    else:
                        prepayment[
                            loan_id
                        ] = min(
                            prepayment[loan_id],
                            period
                        )

    # ========================================================
    # Chronology classification
    # ========================================================

    all_loans = set(
        first_observed
    )

    chronology = Counter()

    same_month_examples = []

    for loan_id in all_loans:

        default_period = first_90.get(
            loan_id
        )

        prepay_period = prepayment.get(
            loan_id
        )

        if (
            default_period is None
            and prepay_period is None
        ):
            chronology[
                "CENSOR_NO_EVENT"
            ] += 1

        elif (
            default_period is not None
            and prepay_period is None
        ):
            chronology[
                "DEFAULT_ONLY"
            ] += 1

        elif (
            default_period is None
            and prepay_period is not None
        ):
            chronology[
                "PREPAY_ONLY"
            ] += 1

        else:

            if default_period < prepay_period:

                chronology[
                    "DEFAULT_BEFORE_PREPAY"
                ] += 1

            elif prepay_period < default_period:

                chronology[
                    "PREPAY_BEFORE_DEFAULT"
                ] += 1

            else:

                chronology[
                    "DEFAULT_AND_PREPAY_SAME_MONTH"
                ] += 1

                if len(same_month_examples) < 10:

                    same_month_examples.append(
                        (
                            loan_id,
                            default_period
                        )
                    )

    result = {
        "vintage": vintage,
        "loans_observed": len(all_loans),

        "censor_no_event":
            chronology["CENSOR_NO_EVENT"],

        "default_only":
            chronology["DEFAULT_ONLY"],

        "prepay_only":
            chronology["PREPAY_ONLY"],

        "default_before_prepay":
            chronology["DEFAULT_BEFORE_PREPAY"],

        "prepay_before_default":
            chronology["PREPAY_BEFORE_DEFAULT"],

        "same_month":
            chronology[
                "DEFAULT_AND_PREPAY_SAME_MONTH"
            ]
    }

    return (
        result,
        chronology,
        same_month_examples
    )


# ============================================================
# 6. SAVE AUDIT SUMMARY
# ============================================================

def save_summary(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "event_chronology_audit.csv"
    )

    fieldnames = [
        "vintage",
        "loans_observed",
        "censor_no_event",
        "default_only",
        "prepay_only",
        "default_before_prepay",
        "prepay_before_default",
        "same_month"
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nAudit summary saved to:\n"
        f"{output_file}"
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "EVENT CHRONOLOGY AUDIT\n"
    )

    columns = read_performance_header()

    loan_id_index = find_column_index(
        columns,
        "LOAN IDENTIFIER"
    )

    period_index = find_column_index(
        columns,
        "PERIOD"
    )

    delinquency_index = find_column_index(
        columns,
        "CURRENT LOAN DELINQUENCY STATUS"
    )

    zero_balance_index = find_column_index(
        columns,
        "ZERO BALANCE CODE"
    )

    zip_files = sorted(
        RAW_DIR.glob("sample_*.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            f"No raw ZIP files found in {RAW_DIR}"
        )

    print(
        f"Vintages found: {len(zip_files)}\n"
    )

    results = []

    overall = Counter()

    total_loans = 0

    all_same_month_examples = []

    for zip_path in zip_files:

        (
            result,
            chronology,
            same_month_examples
        ) = audit_vintage(
            zip_path,
            loan_id_index,
            period_index,
            delinquency_index,
            zero_balance_index
        )

        results.append(result)

        overall.update(
            chronology
        )

        total_loans += result[
            "loans_observed"
        ]

        all_same_month_examples.extend(
            same_month_examples
        )

        print("=" * 70)
        print(
            f"VINTAGE {result['vintage']}"
        )

        print(
            f"Loans observed         : "
            f"{result['loans_observed']:,}"
        )

        print(
            f"No event               : "
            f"{result['censor_no_event']:,}"
        )

        print(
            f"Default only           : "
            f"{result['default_only']:,}"
        )

        print(
            f"Prepay only            : "
            f"{result['prepay_only']:,}"
        )

        print(
            f"Default before prepay  : "
            f"{result['default_before_prepay']:,}"
        )

        print(
            f"Prepay before default  : "
            f"{result['prepay_before_default']:,}"
        )

        print(
            f"Same-month default/pay : "
            f"{result['same_month']:,}"
        )

    # ========================================================
    # Overall summary
    # ========================================================

    print("\n" + "=" * 70)
    print("OVERALL EVENT CHRONOLOGY AUDIT")
    print("=" * 70)

    print(
        f"Loans observed        : "
        f"{total_loans:,}"
    )

    print(
        f"No event              : "
        f"{overall['CENSOR_NO_EVENT']:,}"
    )

    print(
        f"Default only          : "
        f"{overall['DEFAULT_ONLY']:,}"
    )

    print(
        f"Prepay only           : "
        f"{overall['PREPAY_ONLY']:,}"
    )

    print(
        f"Default before prepay : "
        f"{overall['DEFAULT_BEFORE_PREPAY']:,}"
    )

    print(
        f"Prepay before default : "
        f"{overall['PREPAY_BEFORE_DEFAULT']:,}"
    )

    print(
        f"Same month            : "
        f"{overall['DEFAULT_AND_PREPAY_SAME_MONTH']:,}"
    )

    classified_total = sum(
        overall.values()
    )

    print(
        f"\nClassification total  : "
        f"{classified_total:,}"
    )

    print(
        f"Matches loans observed: "
        f"{classified_total == total_loans}"
    )

    if all_same_month_examples:

        print(
            "\nExamples of same-month "
            "Default and Prepayment:"
        )

        for loan_id, period in (
            all_same_month_examples[:10]
        ):
            print(
                f"  {loan_id} | {period}"
            )

    save_summary(results)

    print(
        "\nChronology audit complete. "
        "Final event coding has NOT yet been applied."
    )


if __name__ == "__main__":
    main()