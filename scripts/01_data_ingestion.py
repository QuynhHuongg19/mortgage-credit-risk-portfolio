from pathlib import Path
from zipfile import ZipFile
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
# 2. EXPECTED DATA STRUCTURE
# ============================================================

EXPECTED_ORIG_COLUMNS = 31
EXPECTED_PERF_COLUMNS = 35


# ============================================================
# 3. READ OFFICIAL FREDDIE MAC HEADERS
# ============================================================

def read_header_files():
    """
    Read the official Freddie Mac header files stored inside
    file_headers_july_2026.zip.

    Returns
    -------
    orig_columns : list
        Column names for origination data.

    perf_columns : list
        Column names for monthly performance data.
    """

    if not HEADER_ZIP.exists():
        raise FileNotFoundError(
            f"Header ZIP not found: {HEADER_ZIP}"
        )

    with ZipFile(HEADER_ZIP, "r") as zip_file:

        names = zip_file.namelist()

        print("Official header files found:")

        for name in names:
            print(f"  - {name}")

        orig_header_file = next(
            name for name in names
            if "orig" in name.lower()
        )

        perf_header_file = next(
            name for name in names
            if "perf" in name.lower()
        )

        with zip_file.open(orig_header_file) as file:
            orig_line = file.readline().decode(
                "utf-8-sig",
                errors="replace"
            ).strip()

        with zip_file.open(perf_header_file) as file:
            perf_line = file.readline().decode(
                "utf-8-sig",
                errors="replace"
            ).strip()

    orig_columns = orig_line.split("|")
    perf_columns = perf_line.split("|")

    return orig_columns, perf_columns


# ============================================================
# 4. COUNT RECORDS WITHOUT EXTRACTING ZIP FILES
# ============================================================

def count_records(zip_file, member_name):
    """
    Count records directly inside a ZIP member.

    The function reads the file line by line so the entire
    dataset does not need to be loaded into memory.
    """

    count = 0

    with zip_file.open(member_name) as file:

        for _ in file:
            count += 1

    return count


# ============================================================
# 5. AUDIT ONE VINTAGE
# ============================================================

def audit_vintage(zip_path, orig_columns, perf_columns):
    """
    Validate one Freddie Mac sample vintage.
    """

    vintage = zip_path.stem.split("_")[-1]

    expected_orig = f"sample_orig_{vintage}.txt"
    expected_perf = f"sample_perf_{vintage}.txt"

    result = {
        "vintage": vintage,
        "orig_records": 0,
        "perf_records": 0,
        "orig_columns": 0,
        "perf_columns": 0,
        "status": "PASS"
    }

    print("=" * 72)
    print(f"VINTAGE: {vintage}")

    with ZipFile(zip_path, "r") as zip_file:

        members = zip_file.namelist()

        if expected_orig not in members:
            raise ValueError(
                f"{expected_orig} not found in {zip_path.name}"
            )

        if expected_perf not in members:
            raise ValueError(
                f"{expected_perf} not found in {zip_path.name}"
            )

        # ----------------------------------------------------
        # Check Origination structure
        # ----------------------------------------------------

        with zip_file.open(expected_orig) as file:

            first_orig_line = file.readline().decode(
                "utf-8",
                errors="replace"
            ).rstrip("\r\n")

        orig_values = first_orig_line.split("|")

        # ----------------------------------------------------
        # Check Performance structure
        # ----------------------------------------------------

        with zip_file.open(expected_perf) as file:

            first_perf_line = file.readline().decode(
                "utf-8",
                errors="replace"
            ).rstrip("\r\n")

        perf_values = first_perf_line.split("|")

        result["orig_columns"] = len(orig_values)
        result["perf_columns"] = len(perf_values)

        # ----------------------------------------------------
        # Validate schema
        # ----------------------------------------------------

        if len(orig_values) != len(orig_columns):
            result["status"] = "FAIL"

        if len(perf_values) != len(perf_columns):
            result["status"] = "FAIL"

        if len(orig_values) != EXPECTED_ORIG_COLUMNS:
            result["status"] = "FAIL"

        if len(perf_values) != EXPECTED_PERF_COLUMNS:
            result["status"] = "FAIL"

        # ----------------------------------------------------
        # Count records
        # ----------------------------------------------------

        result["orig_records"] = count_records(
            zip_file,
            expected_orig
        )

        result["perf_records"] = count_records(
            zip_file,
            expected_perf
        )

    print(f"Origination records : {result['orig_records']:,}")
    print(f"Performance records : {result['perf_records']:,}")
    print(f"Origination columns : {result['orig_columns']}")
    print(f"Performance columns : {result['perf_columns']}")
    print(f"Schema status       : {result['status']}")

    return result


# ============================================================
# 6. SAVE AUDIT REPORT
# ============================================================

def save_audit_report(results):
    """
    Save the audit summary as a CSV file.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "data_ingestion_audit.csv"

    fieldnames = [
        "vintage",
        "orig_records",
        "perf_records",
        "orig_columns",
        "perf_columns",
        "status"
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

    print("\nAudit report saved to:")
    print(output_file)


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print("\nMORTGAGE CREDIT RISK - DATA INGESTION AUDIT\n")

    # --------------------------------------------------------
    # Load official Freddie Mac schema
    # --------------------------------------------------------

    orig_columns, perf_columns = read_header_files()

    print()
    print(
        f"Official Origination columns: "
        f"{len(orig_columns)}"
    )

    print(
        f"Official Performance columns: "
        f"{len(perf_columns)}"
    )

    if len(orig_columns) != EXPECTED_ORIG_COLUMNS:
        raise ValueError(
            "Unexpected number of Origination header fields."
        )

    if len(perf_columns) != EXPECTED_PERF_COLUMNS:
        raise ValueError(
            "Unexpected number of Performance header fields."
        )

    # --------------------------------------------------------
    # Discover raw ZIP files
    # --------------------------------------------------------

    zip_files = sorted(
        RAW_DIR.glob("sample_*.zip")
    )

    print(f"Raw ZIP files found: {len(zip_files)}\n")

    if not zip_files:
        raise FileNotFoundError(
            f"No sample ZIP files found in {RAW_DIR}"
        )

    # --------------------------------------------------------
    # Audit every vintage
    # --------------------------------------------------------

    results = []

    for zip_path in zip_files:

        result = audit_vintage(
            zip_path,
            orig_columns,
            perf_columns
        )

        results.append(result)

    # --------------------------------------------------------
    # Save audit summary
    # --------------------------------------------------------

    save_audit_report(results)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_orig = sum(
        item["orig_records"]
        for item in results
    )

    total_perf = sum(
        item["perf_records"]
        for item in results
    )

    failed = [
        item
        for item in results
        if item["status"] != "PASS"
    ]

    print("\n" + "=" * 72)
    print("FINAL AUDIT SUMMARY")
    print("=" * 72)

    print(f"Vintages audited      : {len(results)}")
    print(f"Origination records   : {total_orig:,}")
    print(f"Performance records   : {total_perf:,}")
    print(f"Schema failures       : {len(failed)}")

    if failed:
        print("\nAUDIT RESULT: FAILED")
    else:
        print("\nAUDIT RESULT: PASSED")


if __name__ == "__main__":
    main()