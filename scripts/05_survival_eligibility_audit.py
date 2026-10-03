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
# 2. READ OFFICIAL HEADERS
# ============================================================

def read_headers():

    with ZipFile(HEADER_ZIP, "r") as zip_file:

        names = zip_file.namelist()

        orig_file = next(
            name for name in names
            if "orig" in name.lower()
        )

        perf_file = next(
            name for name in names
            if "perf" in name.lower()
        )

        with zip_file.open(orig_file) as file:
            orig_line = file.readline().decode(
                "utf-8-sig",
                errors="replace"
            ).rstrip("\r\n")

        with zip_file.open(perf_file) as file:
            perf_line = file.readline().decode(
                "utf-8-sig",
                errors="replace"
            ).rstrip("\r\n")

    return (
        orig_line.split("|"),
        perf_line.split("|")
    )


# ============================================================
# 3. COLUMN LOOKUP
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

def period_to_int(value):

    value = value.strip()

    if len(value) == 6 and value.isdigit():
        return int(value)

    return None


# ============================================================
# 5. AUDIT ONE VINTAGE
# ============================================================

def audit_vintage(
    zip_path,
    orig_loan_index,
    perf_loan_index,
    period_index,
    loan_age_index,
    delinquency_index,
    zero_balance_index
):

    vintage = zip_path.stem.split("_")[-1]

    orig_file = f"sample_orig_{vintage}.txt"
    perf_file = f"sample_perf_{vintage}.txt"

    origination_ids = set()

    performance_ids = set()

    first_period = {}
    last_period = {}

    first_90 = {}
    prepayment = {}

    other_termination = {}

    same_month_details = []

    # --------------------------------------------------------
    # Origination IDs
    # --------------------------------------------------------

    with ZipFile(zip_path, "r") as zip_file:

        with zip_file.open(orig_file) as file:

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
                    orig_loan_index
                ].strip()

                if loan_id:
                    origination_ids.add(
                        loan_id
                    )

        # ----------------------------------------------------
        # Performance history
        # ----------------------------------------------------

        with zip_file.open(perf_file) as file:

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
                    perf_loan_index
                ].strip()

                period = period_to_int(
                    values[period_index]
                )

                loan_age = values[
                    loan_age_index
                ].strip()

                delinquency_raw = values[
                    delinquency_index
                ].strip()

                zero_balance = values[
                    zero_balance_index
                ].strip()

                if not loan_id:
                    continue

                performance_ids.add(
                    loan_id
                )

                if period is not None:

                    if loan_id not in first_period:
                        first_period[
                            loan_id
                        ] = period
                    else:
                        first_period[
                            loan_id
                        ] = min(
                            first_period[loan_id],
                            period
                        )

                    if loan_id not in last_period:
                        last_period[
                            loan_id
                        ] = period
                    else:
                        last_period[
                            loan_id
                        ] = max(
                            last_period[loan_id],
                            period
                        )

                # --------------------------------------------
                # First 90+ DPD
                # --------------------------------------------

                if delinquency_raw.isdigit():

                    delinquency = int(
                        delinquency_raw
                    )

                    if (
                        delinquency >= 3
                        and period is not None
                    ):

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
                # Prepayment
                # --------------------------------------------

                if (
                    zero_balance == "01"
                    and period is not None
                ):

                    prepayment[
                        loan_id
                    ] = period

                # --------------------------------------------
                # Other termination
                # --------------------------------------------

                if (
                    zero_balance
                    in {"15", "16", "96"}
                    and period is not None
                ):

                    other_termination[
                        loan_id
                    ] = (
                        period,
                        zero_balance
                    )

                # --------------------------------------------
                # Same-month detail
                # We collect this later after first_90 is known.
                # --------------------------------------------

    # ========================================================
    # Missing / unexpected IDs
    # ========================================================

    missing_performance = (
        origination_ids
        - performance_ids
    )

    performance_without_origination = (
        performance_ids
        - origination_ids
    )

    # ========================================================
    # Same-month Default + Prepayment
    # ========================================================

    same_month_ids = []

    for loan_id in (
        set(first_90)
        & set(prepayment)
    ):

        if (
            first_90[loan_id]
            == prepayment[loan_id]
        ):
            same_month_ids.append(
                loan_id
            )

    # Re-read only to obtain details for same-month cases
    same_month_set = set(
        same_month_ids
    )

    if same_month_set:

        with ZipFile(zip_path, "r") as zip_file:

            with zip_file.open(perf_file) as file:

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
                        perf_loan_index
                    ].strip()

                    if loan_id not in same_month_set:
                        continue

                    period = period_to_int(
                        values[period_index]
                    )

                    if (
                        period
                        != first_90[loan_id]
                    ):
                        continue

                    same_month_details.append(
                        {
                            "vintage": vintage,
                            "loan_id": loan_id,
                            "period": period,
                            "loan_age": values[
                                loan_age_index
                            ].strip(),
                            "delinquency_status":
                                values[
                                    delinquency_index
                                ].strip(),
                            "zero_balance_code":
                                values[
                                    zero_balance_index
                                ].strip()
                        }
                    )

    # ========================================================
    # Other termination chronology
    # ========================================================

    other_counts = Counter()

    for loan_id, (
        termination_period,
        termination_code
    ) in other_termination.items():

        default_period = first_90.get(
            loan_id
        )

        if default_period is None:

            other_counts[
                f"{termination_code}_NO_DEFAULT"
            ] += 1

        elif default_period < termination_period:

            other_counts[
                f"{termination_code}_AFTER_DEFAULT"
            ] += 1

        elif default_period == termination_period:

            other_counts[
                f"{termination_code}_SAME_MONTH_DEFAULT"
            ] += 1

        else:

            other_counts[
                f"{termination_code}_BEFORE_DEFAULT"
            ] += 1

    result = {
        "vintage": vintage,
        "origination_loans":
            len(origination_ids),
        "performance_loans":
            len(performance_ids),
        "missing_performance":
            len(missing_performance),
        "performance_without_origination":
            len(performance_without_origination),
        "same_month_default_prepay":
            len(same_month_ids),
        "other_termination_loans":
            len(other_termination)
    }

    return (
        result,
        sorted(missing_performance),
        sorted(
            performance_without_origination
        ),
        same_month_details,
        other_counts
    )


# ============================================================
# 6. SAVE SAME-MONTH DETAILS
# ============================================================

def save_same_month_details(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "same_month_default_prepay.csv"
    )

    fieldnames = [
        "vintage",
        "loan_id",
        "period",
        "loan_age",
        "delinquency_status",
        "zero_balance_code"
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
        writer.writerows(rows)

    return output_file


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "SURVIVAL ELIGIBILITY AUDIT\n"
    )

    orig_columns, perf_columns = (
        read_headers()
    )

    orig_loan_index = find_column_index(
        orig_columns,
        "LOAN IDENTIFIER"
    )

    perf_loan_index = find_column_index(
        perf_columns,
        "LOAN IDENTIFIER"
    )

    period_index = find_column_index(
        perf_columns,
        "PERIOD"
    )

    loan_age_index = find_column_index(
        perf_columns,
        "LOAN AGE"
    )

    delinquency_index = find_column_index(
        perf_columns,
        "CURRENT LOAN DELINQUENCY STATUS"
    )

    zero_balance_index = find_column_index(
        perf_columns,
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

    total_orig = 0
    total_perf = 0

    all_missing = []
    all_unexpected = []
    all_same_month = []

    overall_other = Counter()

    for zip_path in zip_files:

        (
            result,
            missing,
            unexpected,
            same_month,
            other_counts
        ) = audit_vintage(
            zip_path,
            orig_loan_index,
            perf_loan_index,
            period_index,
            loan_age_index,
            delinquency_index,
            zero_balance_index
        )

        total_orig += result[
            "origination_loans"
        ]

        total_perf += result[
            "performance_loans"
        ]

        all_missing.extend(
            [
                (
                    result["vintage"],
                    loan_id
                )
                for loan_id in missing
            ]
        )

        all_unexpected.extend(
            [
                (
                    result["vintage"],
                    loan_id
                )
                for loan_id in unexpected
            ]
        )

        all_same_month.extend(
            same_month
        )

        overall_other.update(
            other_counts
        )

        print("=" * 70)

        print(
            f"VINTAGE {result['vintage']}"
        )

        print(
            "Origination loans          : "
            f"{result['origination_loans']:,}"
        )

        print(
            "Performance loans          : "
            f"{result['performance_loans']:,}"
        )

        print(
            "Missing performance        : "
            f"{result['missing_performance']:,}"
        )

        print(
            "Performance without orig   : "
            f"{result['performance_without_origination']:,}"
        )

        print(
            "Same-month default/prepay  : "
            f"{result['same_month_default_prepay']:,}"
        )

        print(
            "Other termination loans    : "
            f"{result['other_termination_loans']:,}"
        )

    # ========================================================
    # OVERALL
    # ========================================================

    print("\n" + "=" * 70)
    print("OVERALL SURVIVAL ELIGIBILITY AUDIT")
    print("=" * 70)

    print(
        f"Origination loans        : "
        f"{total_orig:,}"
    )

    print(
        f"Performance loans        : "
        f"{total_perf:,}"
    )

    print(
        f"Missing performance      : "
        f"{len(all_missing):,}"
    )

    print(
        f"Performance without orig : "
        f"{len(all_unexpected):,}"
    )

    print(
        f"Same-month cases         : "
        f"{len(all_same_month):,}"
    )

    # --------------------------------------------------------
    # Missing performance details
    # --------------------------------------------------------

    if all_missing:

        print(
            "\nOrigination loans with "
            "no performance history:"
        )

        for vintage, loan_id in all_missing:
            print(
                f"  {vintage} | {loan_id}"
            )

    # --------------------------------------------------------
    # Unexpected performance IDs
    # --------------------------------------------------------

    if all_unexpected:

        print(
            "\nPerformance loans with "
            "no matching origination:"
        )

        for vintage, loan_id in all_unexpected:
            print(
                f"  {vintage} | {loan_id}"
            )

    # --------------------------------------------------------
    # Other termination chronology
    # --------------------------------------------------------

    print(
        "\nOther termination chronology "
        "(codes 15, 16, 96):"
    )

    for label, count in sorted(
        overall_other.items()
    ):
        print(
            f"  {label:<30}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Same-month cases
    # --------------------------------------------------------

    print(
        "\nSame-month Default + Prepayment "
        "status distribution:"
    )

    same_month_status = Counter(
        row["delinquency_status"]
        for row in all_same_month
    )

    for status, count in sorted(
        same_month_status.items()
    ):
        print(
            f"  Status {status:>4}: "
            f"{count:,}"
        )

    output_file = save_same_month_details(
        all_same_month
    )

    print(
        f"\nSame-month details saved to:\n"
        f"{output_file}"
    )

    print(
        "\nEligibility audit complete. "
        "Final survival cohort has NOT "
        "yet been created."
    )


if __name__ == "__main__":
    main()