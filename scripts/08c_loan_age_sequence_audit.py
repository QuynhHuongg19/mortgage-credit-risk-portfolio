from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import csv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "loan_age_sequence_audit.csv"
)


# ============================================================
# 2. HELPERS
# ============================================================

def parse_period(value):
    """
    Parse YYYYMM into a continuous month index.
    """

    value = value.strip()

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

    return year * 12 + month - 1


def parse_int(value):

    value = value.strip()

    if value == "":
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


# ============================================================
# 3. AUDIT ONE VINTAGE
# ============================================================

def audit_vintage(zip_path):

    vintage = (
        zip_path.stem
        .split("_")[-1]
    )

    performance_name = (
        f"sample_perf_{vintage}.txt"
    )

    total_rows = 0
    valid_rows = 0

    same_loan_transitions = 0

    exact_progression = 0
    calendar_gap_consistent = 0

    same_period_same_age = 0

    age_reset = 0
    age_decrease = 0

    period_decrease = 0

    inconsistent_positive_jump = 0

    invalid_period = 0
    invalid_age = 0

    period_gap_distribution = Counter()
    age_gap_distribution = Counter()
    difference_distribution = Counter()

    examples = []

    previous_loan = None
    previous_period = None
    previous_age = None
    previous_period_raw = None

    with ZipFile(
        zip_path,
        "r"
    ) as archive:

        if (
            performance_name
            not in archive.namelist()
        ):
            raise FileNotFoundError(
                f"{performance_name} "
                f"not found in "
                f"{zip_path.name}"
            )

        with archive.open(
            performance_name
        ) as file:

            for raw_line in file:

                total_rows += 1

                line = raw_line.decode(
                    "utf-8",
                    errors="replace"
                ).rstrip(
                    "\r\n"
                )

                fields = line.split("|")

                # Official performance schema:
                # 0 = LOAN IDENTIFIER
                # 1 = PERIOD
                # 4 = LOAN AGE
                if len(fields) <= 4:
                    continue

                loan_id = (
                    fields[0].strip()
                )

                period_raw = (
                    fields[1].strip()
                )

                age_raw = (
                    fields[4].strip()
                )

                period = parse_period(
                    period_raw
                )

                age = parse_int(
                    age_raw
                )

                if period is None:
                    invalid_period += 1

                if age is None:
                    invalid_age += 1

                if (
                    not loan_id
                    or period is None
                    or age is None
                ):

                    previous_loan = None
                    previous_period = None
                    previous_age = None
                    previous_period_raw = None

                    continue

                valid_rows += 1

                if (
                    loan_id
                    == previous_loan
                    and previous_period
                    is not None
                    and previous_age
                    is not None
                ):

                    same_loan_transitions += 1

                    period_gap = (
                        period
                        - previous_period
                    )

                    age_gap = (
                        age
                        - previous_age
                    )

                    difference = (
                        age_gap
                        - period_gap
                    )

                    period_gap_distribution[
                        period_gap
                    ] += 1

                    age_gap_distribution[
                        age_gap
                    ] += 1

                    difference_distribution[
                        difference
                    ] += 1

                    # ------------------------------------
                    # Exact monthly progression
                    # ------------------------------------

                    if (
                        period_gap == 1
                        and age_gap == 1
                    ):

                        exact_progression += 1

                    # ------------------------------------
                    # Calendar gaps where Loan Age moves
                    # by exactly the same number of months
                    # ------------------------------------

                    elif (
                        period_gap > 1
                        and age_gap
                        == period_gap
                    ):

                        calendar_gap_consistent += 1

                    # ------------------------------------
                    # Duplicate period / same age
                    # ------------------------------------

                    elif (
                        period_gap == 0
                        and age_gap == 0
                    ):

                        same_period_same_age += 1

                    # ------------------------------------
                    # Period goes backwards
                    # ------------------------------------

                    elif period_gap < 0:

                        period_decrease += 1

                    # ------------------------------------
                    # Loan Age decreases / resets
                    # ------------------------------------

                    elif age_gap < 0:

                        age_decrease += 1

                        if age == 0:
                            age_reset += 1

                    # ------------------------------------
                    # Positive but inconsistent movement
                    # ------------------------------------

                    else:

                        inconsistent_positive_jump += 1

                    if (
                        difference != 0
                        and len(examples) < 20
                    ):

                        examples.append(
                            {
                                "loan_id":
                                    loan_id,

                                "previous_period":
                                    previous_period_raw,

                                "current_period":
                                    period_raw,

                                "previous_age":
                                    previous_age,

                                "current_age":
                                    age,

                                "period_gap":
                                    period_gap,

                                "age_gap":
                                    age_gap,

                                "difference":
                                    difference,
                            }
                        )

                previous_loan = loan_id
                previous_period = period
                previous_age = age
                previous_period_raw = (
                    period_raw
                )

    return {
        "vintage":
            vintage,

        "total_rows":
            total_rows,

        "valid_rows":
            valid_rows,

        "same_loan_transitions":
            same_loan_transitions,

        "exact_progression":
            exact_progression,

        "calendar_gap_consistent":
            calendar_gap_consistent,

        "same_period_same_age":
            same_period_same_age,

        "age_reset":
            age_reset,

        "age_decrease":
            age_decrease,

        "period_decrease":
            period_decrease,

        "inconsistent_positive_jump":
            inconsistent_positive_jump,

        "invalid_period":
            invalid_period,

        "invalid_age":
            invalid_age,

        "period_gap_distribution":
            period_gap_distribution,

        "age_gap_distribution":
            age_gap_distribution,

        "difference_distribution":
            difference_distribution,

        "examples":
            examples,
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():

    zip_files = sorted(
        RAW_DIR.glob(
            "sample_*.zip"
        )
    )

    if not zip_files:

        raise FileNotFoundError(
            f"No raw ZIP files found:\n"
            f"{RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    overall_difference = Counter()

    print(
        "\nMORTGAGE CREDIT RISK - "
        "LOAN AGE SEQUENCE AUDIT\n"
    )

    for zip_path in zip_files:

        vintage = (
            zip_path.stem
            .split("_")[-1]
        )

        print(
            f"Auditing vintage "
            f"{vintage}..."
        )

        result = audit_vintage(
            zip_path
        )

        results.append(
            result
        )

        overall_difference.update(
            result[
                "difference_distribution"
            ]
        )

    # ========================================================
    # 5. SAVE SUMMARY
    # ========================================================

    fieldnames = [
        "vintage",
        "total_rows",
        "valid_rows",
        "same_loan_transitions",
        "exact_progression",
        "calendar_gap_consistent",
        "same_period_same_age",
        "age_reset",
        "age_decrease",
        "period_decrease",
        "inconsistent_positive_jump",
        "invalid_period",
        "invalid_age",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    key:
                        result[key]
                    for key in fieldnames
                }
            )

    # ========================================================
    # 6. PRINT VINTAGE SUMMARY
    # ========================================================

    print("\n" + "=" * 118)

    print(
        f"{'Vintage':<8}"
        f"{'Rows':>12}"
        f"{'Transitions':>14}"
        f"{'Exact':>14}"
        f"{'Gap OK':>11}"
        f"{'Dup':>8}"
        f"{'AgeDec':>9}"
        f"{'Reset':>8}"
        f"{'PerDec':>9}"
        f"{'Other':>10}"
    )

    print("=" * 118)

    for r in results:

        print(
            f"{r['vintage']:<8}"
            f"{r['total_rows']:>12,}"
            f"{r['same_loan_transitions']:>14,}"
            f"{r['exact_progression']:>14,}"
            f"{r['calendar_gap_consistent']:>11,}"
            f"{r['same_period_same_age']:>8,}"
            f"{r['age_decrease']:>9,}"
            f"{r['age_reset']:>8,}"
            f"{r['period_decrease']:>9,}"
            f"{r['inconsistent_positive_jump']:>10,}"
        )

    # ========================================================
    # 7. OVERALL TOTALS
    # ========================================================

    total_rows = sum(
        r["total_rows"]
        for r in results
    )

    total_transitions = sum(
        r["same_loan_transitions"]
        for r in results
    )

    total_exact = sum(
        r["exact_progression"]
        for r in results
    )

    total_gap_ok = sum(
        r["calendar_gap_consistent"]
        for r in results
    )

    total_duplicates = sum(
        r["same_period_same_age"]
        for r in results
    )

    total_age_decrease = sum(
        r["age_decrease"]
        for r in results
    )

    total_age_reset = sum(
        r["age_reset"]
        for r in results
    )

    total_period_decrease = sum(
        r["period_decrease"]
        for r in results
    )

    total_other = sum(
        r["inconsistent_positive_jump"]
        for r in results
    )

    total_invalid_period = sum(
        r["invalid_period"]
        for r in results
    )

    total_invalid_age = sum(
        r["invalid_age"]
        for r in results
    )

    print("\n" + "=" * 78)

    print(
        "OVERALL LOAN AGE SEQUENCE AUDIT"
    )

    print("=" * 78)

    print(
        f"Performance rows              : "
        f"{total_rows:,}"
    )

    print(
        f"Same-loan transitions         : "
        f"{total_transitions:,}"
    )

    print(
        f"Exact monthly progression     : "
        f"{total_exact:,}"
    )

    print(
        f"Consistent calendar gaps      : "
        f"{total_gap_ok:,}"
    )

    print(
        f"Same period / same age        : "
        f"{total_duplicates:,}"
    )

    print(
        f"Loan age decreases            : "
        f"{total_age_decrease:,}"
    )

    print(
        f"Loan age resets to zero       : "
        f"{total_age_reset:,}"
    )

    print(
        f"Performance period decreases  : "
        f"{total_period_decrease:,}"
    )

    print(
        f"Other inconsistent transitions: "
        f"{total_other:,}"
    )

    print(
        f"Invalid PERIOD values         : "
        f"{total_invalid_period:,}"
    )

    print(
        f"Invalid LOAN AGE values       : "
        f"{total_invalid_age:,}"
    )

    if total_transitions:

        consistent = (
            total_exact
            + total_gap_ok
            + total_duplicates
        )

        print(
            f"\nStructurally consistent       : "
            f"{consistent:,} "
            f"({100 * consistent / total_transitions:.4f}%)"
        )

    # ========================================================
    # 8. DIFFERENCE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 78)

    print(
        "AGE GAP - CALENDAR GAP DISTRIBUTION"
    )

    print("=" * 78)

    for difference, count in (
        overall_difference.most_common(
            20
        )
    ):

        print(
            f"Difference {difference:>5}: "
            f"{count:>12,}"
        )

    # ========================================================
    # 9. EXAMPLES
    # ========================================================

    example_count = 0

    print("\n" + "=" * 78)

    print(
        "EXAMPLES OF NONZERO DIFFERENCE"
    )

    print("=" * 78)

    for result in results:

        for example in (
            result["examples"]
        ):

            if example_count >= 20:
                break

            print(
                f"{result['vintage']} | "
                f"{example}"
            )

            example_count += 1

        if example_count >= 20:
            break

    print(
        f"\nAudit saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nLoan age sequence audit complete."
    )


if __name__ == "__main__":
    main()