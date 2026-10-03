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
    """
    Read the official Freddie Mac monthly performance header.
    """

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
# 4. AUDIT TERMINATION EVENTS FOR ONE VINTAGE
# ============================================================

def audit_vintage(
    zip_path,
    loan_id_index,
    period_index,
    zero_balance_index
):
    """
    Inspect termination records at loan level.

    For every loan:
    - count nonblank ZERO BALANCE CODE observations
    - record unique termination codes
    - retain termination periods
    """

    vintage = zip_path.stem.split("_")[-1]

    performance_file = (
        f"sample_perf_{vintage}.txt"
    )

    loan_codes = defaultdict(list)

    total_rows = 0
    rows_with_code = 0

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

                code = values[
                    zero_balance_index
                ].strip()

                if code != "":

                    rows_with_code += 1

                    loan_codes[loan_id].append(
                        (period, code)
                    )

    # --------------------------------------------------------
    # Loan-level diagnostics
    # --------------------------------------------------------

    loans_with_code = len(loan_codes)

    repeated_code_loans = 0
    conflicting_code_loans = 0

    terminal_code_counts = Counter()

    examples_repeated = []
    examples_conflicting = []

    for loan_id, observations in loan_codes.items():

        if len(observations) > 1:

            repeated_code_loans += 1

            if len(examples_repeated) < 5:
                examples_repeated.append(
                    (loan_id, observations)
                )

        unique_codes = {
            code
            for _, code in observations
        }

        if len(unique_codes) > 1:

            conflicting_code_loans += 1

            if len(examples_conflicting) < 5:
                examples_conflicting.append(
                    (loan_id, observations)
                )

        # Last reported nonblank termination observation
        final_period, final_code = max(
            observations,
            key=lambda item: item[0]
        )

        terminal_code_counts[
            final_code
        ] += 1

    result = {
        "vintage": vintage,
        "performance_rows": total_rows,
        "rows_with_zero_balance_code": rows_with_code,
        "loans_with_zero_balance_code": loans_with_code,
        "loans_with_repeated_code": repeated_code_loans,
        "loans_with_conflicting_codes": conflicting_code_loans
    }

    return (
        result,
        terminal_code_counts,
        examples_repeated,
        examples_conflicting
    )


# ============================================================
# 5. SAVE SUMMARY
# ============================================================

def save_summary(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "zero_balance_loan_audit.csv"
    )

    fieldnames = [
        "vintage",
        "performance_rows",
        "rows_with_zero_balance_code",
        "loans_with_zero_balance_code",
        "loans_with_repeated_code",
        "loans_with_conflicting_codes"
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
# 6. MAIN
# ============================================================

def main():

    print(
        "\nMORTGAGE CREDIT RISK - "
        "LOAN-LEVEL TERMINATION AUDIT\n"
    )

    # --------------------------------------------------------
    # Official schema
    # --------------------------------------------------------

    columns = read_performance_header()

    loan_id_index = find_column_index(
        columns,
        "LOAN IDENTIFIER"
    )

    period_index = find_column_index(
        columns,
        "PERIOD"
    )

    zero_balance_index = find_column_index(
        columns,
        "ZERO BALANCE CODE"
    )

    print(
        f"LOAN IDENTIFIER position : "
        f"{loan_id_index + 1}"
    )

    print(
        f"PERIOD position          : "
        f"{period_index + 1}"
    )

    print(
        f"ZERO BALANCE CODE position: "
        f"{zero_balance_index + 1}"
    )

    # --------------------------------------------------------
    # Raw files
    # --------------------------------------------------------

    zip_files = sorted(
        RAW_DIR.glob("sample_*.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP files found in {RAW_DIR}"
        )

    print(
        f"Vintages found: "
        f"{len(zip_files)}\n"
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    results = []

    overall_terminal_codes = Counter()

    total_rows_with_code = 0
    total_loans_with_code = 0
    total_repeated = 0
    total_conflicting = 0

    for zip_path in zip_files:

        (
            result,
            terminal_codes,
            repeated_examples,
            conflicting_examples
        ) = audit_vintage(
            zip_path,
            loan_id_index,
            period_index,
            zero_balance_index
        )

        results.append(result)

        overall_terminal_codes.update(
            terminal_codes
        )

        total_rows_with_code += (
            result[
                "rows_with_zero_balance_code"
            ]
        )

        total_loans_with_code += (
            result[
                "loans_with_zero_balance_code"
            ]
        )

        total_repeated += (
            result[
                "loans_with_repeated_code"
            ]
        )

        total_conflicting += (
            result[
                "loans_with_conflicting_codes"
            ]
        )

        print("=" * 70)
        print(
            f"VINTAGE {result['vintage']}"
        )

        print(
            "Rows with code          : "
            f"{result['rows_with_zero_balance_code']:,}"
        )

        print(
            "Loans with code         : "
            f"{result['loans_with_zero_balance_code']:,}"
        )

        print(
            "Loans with repeated code: "
            f"{result['loans_with_repeated_code']:,}"
        )

        print(
            "Conflicting-code loans  : "
            f"{result['loans_with_conflicting_codes']:,}"
        )

        if conflicting_examples:

            print(
                "\nExample conflicting loans:"
            )

            for example in conflicting_examples:
                print(f"  {example}")

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL LOAN-LEVEL TERMINATION AUDIT")
    print("=" * 70)

    print(
        f"Rows with nonblank code : "
        f"{total_rows_with_code:,}"
    )

    print(
        f"Loans with code         : "
        f"{total_loans_with_code:,}"
    )

    print(
        f"Loans with repeated code: "
        f"{total_repeated:,}"
    )

    print(
        f"Conflicting-code loans  : "
        f"{total_conflicting:,}"
    )

    print(
        "\nTerminal code distribution "
        "(one final code per terminated loan):"
    )

    for code, count in sorted(
        overall_terminal_codes.items()
    ):
        print(
            f"  {code}: {count:,}"
        )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    save_summary(results)

    print(
        "\nAudit complete. "
        "Survival event mapping has NOT yet "
        "been applied."
    )


if __name__ == "__main__":
    main()