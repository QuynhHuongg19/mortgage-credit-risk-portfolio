from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
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
# 3. FIND COLUMN POSITION
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
# 4. CONVERT DELINQUENCY STATUS
# ============================================================

def parse_delinquency(status):
    """
    Convert numeric delinquency status to integer.

    Freddie Mac may also contain non-numeric status values.
    Those values are retained separately rather than forced
    into a numeric delinquency bucket.
    """

    status = status.strip()

    if status == "":
        return None

    if status.isdigit():
        return int(status)

    return None


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

    status_counts = Counter()

    # Maximum numeric delinquency reached by each loan
    max_delinquency = {}

    # First month each loan reaches selected thresholds
    first_30 = {}
    first_60 = {}
    first_90 = {}
    first_120 = {}

    # Termination information
    termination = {}

    total_rows = 0

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

                total_rows += 1

                loan_id = values[
                    loan_id_index
                ].strip()

                period = values[
                    period_index
                ].strip()

                status = values[
                    delinquency_index
                ].strip()

                zero_balance_code = values[
                    zero_balance_index
                ].strip()

                # --------------------------------------------
                # Raw status distribution
                # --------------------------------------------

                display_status = (
                    status
                    if status != ""
                    else "<BLANK>"
                )

                status_counts[
                    display_status
                ] += 1

                # --------------------------------------------
                # Numeric delinquency
                # --------------------------------------------

                delinquency = parse_delinquency(
                    status
                )

                if delinquency is not None:

                    previous_max = (
                        max_delinquency.get(
                            loan_id,
                            -1
                        )
                    )

                    if delinquency > previous_max:
                        max_delinquency[
                            loan_id
                        ] = delinquency

                    # Freddie Mac numeric delinquency status
                    # represents number of payments past due.
                    # Thresholds below are audited only;
                    # no default definition is imposed here.

                    if (
                        delinquency >= 1
                        and loan_id not in first_30
                    ):
                        first_30[loan_id] = period

                    if (
                        delinquency >= 2
                        and loan_id not in first_60
                    ):
                        first_60[loan_id] = period

                    if (
                        delinquency >= 3
                        and loan_id not in first_90
                    ):
                        first_90[loan_id] = period

                    if (
                        delinquency >= 4
                        and loan_id not in first_120
                    ):
                        first_120[loan_id] = period

                # --------------------------------------------
                # Zero-balance termination
                # --------------------------------------------

                if zero_balance_code != "":

                    termination[loan_id] = (
                        period,
                        zero_balance_code
                    )

    # --------------------------------------------------------
    # Loan-level maximum delinquency distribution
    # --------------------------------------------------------

    max_bucket_counts = Counter()

    for value in max_delinquency.values():

        if value == 0:
            bucket = "Never delinquent"

        elif value == 1:
            bucket = "30+ DPD"

        elif value == 2:
            bucket = "60+ DPD"

        elif value == 3:
            bucket = "90+ DPD"

        else:
            bucket = "120+ DPD"

        max_bucket_counts[bucket] += 1

    # --------------------------------------------------------
    # Relationship between serious delinquency and termination
    # --------------------------------------------------------

    termination_after_90 = Counter()
    termination_after_120 = Counter()

    for loan_id in first_90:

        if loan_id in termination:
            _, code = termination[loan_id]
            termination_after_90[code] += 1
        else:
            termination_after_90[
                "<NO TERMINATION CODE>"
            ] += 1

    for loan_id in first_120:

        if loan_id in termination:
            _, code = termination[loan_id]
            termination_after_120[code] += 1
        else:
            termination_after_120[
                "<NO TERMINATION CODE>"
            ] += 1

    summary = {
        "vintage": vintage,
        "performance_rows": total_rows,
        "loans_observed": len(max_delinquency),
        "loans_30_plus": len(first_30),
        "loans_60_plus": len(first_60),
        "loans_90_plus": len(first_90),
        "loans_120_plus": len(first_120)
    }

    return (
        summary,
        status_counts,
        max_bucket_counts,
        termination_after_90,
        termination_after_120
    )


# ============================================================
# 6. SAVE VINTAGE SUMMARY
# ============================================================

def save_summary(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "delinquency_audit.csv"
    )

    fieldnames = [
        "vintage",
        "performance_rows",
        "loans_observed",
        "loans_30_plus",
        "loans_60_plus",
        "loans_90_plus",
        "loans_120_plus"
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
        "DELINQUENCY AUDIT\n"
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

    print(
        "LOAN IDENTIFIER position        : "
        f"{loan_id_index + 1}"
    )

    print(
        "PERIOD position                 : "
        f"{period_index + 1}"
    )

    print(
        "DELINQUENCY STATUS position     : "
        f"{delinquency_index + 1}"
    )

    print(
        "ZERO BALANCE CODE position      : "
        f"{zero_balance_index + 1}"
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

    overall_status = Counter()
    overall_max_buckets = Counter()
    overall_after_90 = Counter()
    overall_after_120 = Counter()

    total_30 = 0
    total_60 = 0
    total_90 = 0
    total_120 = 0

    for zip_path in zip_files:

        (
            summary,
            status_counts,
            max_buckets,
            after_90,
            after_120
        ) = audit_vintage(
            zip_path,
            loan_id_index,
            period_index,
            delinquency_index,
            zero_balance_index
        )

        results.append(summary)

        overall_status.update(
            status_counts
        )

        overall_max_buckets.update(
            max_buckets
        )

        overall_after_90.update(
            after_90
        )

        overall_after_120.update(
            after_120
        )

        total_30 += summary[
            "loans_30_plus"
        ]

        total_60 += summary[
            "loans_60_plus"
        ]

        total_90 += summary[
            "loans_90_plus"
        ]

        total_120 += summary[
            "loans_120_plus"
        ]

        print("=" * 70)
        print(
            f"VINTAGE {summary['vintage']}"
        )

        print(
            f"Loans observed : "
            f"{summary['loans_observed']:,}"
        )

        print(
            f"Ever 30+ DPD   : "
            f"{summary['loans_30_plus']:,}"
        )

        print(
            f"Ever 60+ DPD   : "
            f"{summary['loans_60_plus']:,}"
        )

        print(
            f"Ever 90+ DPD   : "
            f"{summary['loans_90_plus']:,}"
        )

        print(
            f"Ever 120+ DPD  : "
            f"{summary['loans_120_plus']:,}"
        )

    # --------------------------------------------------------
    # Overall audit
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL DELINQUENCY AUDIT")
    print("=" * 70)

    print(
        f"Ever 30+ DPD  : {total_30:,}"
    )

    print(
        f"Ever 60+ DPD  : {total_60:,}"
    )

    print(
        f"Ever 90+ DPD  : {total_90:,}"
    )

    print(
        f"Ever 120+ DPD : {total_120:,}"
    )

    print(
        "\nRaw CURRENT LOAN DELINQUENCY "
        "STATUS distribution:"
    )

    for status, count in sorted(
        overall_status.items()
    ):
        print(
            f"  {status:>10}: {count:,}"
        )

    print(
        "\nMaximum delinquency reached "
        "per loan:"
    )

    bucket_order = [
        "Never delinquent",
        "30+ DPD",
        "60+ DPD",
        "90+ DPD",
        "120+ DPD"
    ]

    for bucket in bucket_order:
        print(
            f"  {bucket:<18}: "
            f"{overall_max_buckets[bucket]:,}"
        )

    print(
        "\nTermination code among loans "
        "that ever reached 90+ DPD:"
    )

    for code, count in sorted(
        overall_after_90.items()
    ):
        print(
            f"  {code:>21}: {count:,}"
        )

    print(
        "\nTermination code among loans "
        "that ever reached 120+ DPD:"
    )

    for code, count in sorted(
        overall_after_120.items()
    ):
        print(
            f"  {code:>21}: {count:,}"
        )

    save_summary(results)

    print(
        "\nAudit complete. "
        "No default definition has been imposed."
    )


if __name__ == "__main__":
    main()