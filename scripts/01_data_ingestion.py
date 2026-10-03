from pathlib import Path
from zipfile import ZipFile


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# 2. BASIC FILE AUDIT
# ============================================================

def audit_zip(zip_path):
    """
    Inspect one Freddie Mac ZIP file without extracting it.
    """

    print("=" * 70)
    print(f"FILE: {zip_path.name}")

    with ZipFile(zip_path, "r") as zip_file:

        file_names = zip_file.namelist()

        print(f"Files inside ZIP: {len(file_names)}")

        for file_name in file_names:
            print(f"  - {file_name}")

        # Inspect each text file
        for file_name in file_names:

            if not file_name.endswith(".txt"):
                continue

            print(f"\nInspecting: {file_name}")

            with zip_file.open(file_name) as file:

                first_line = file.readline().decode(
                    "utf-8",
                    errors="replace"
                ).strip()

                columns = first_line.split("|")

                print(f"Number of columns: {len(columns)}")
                print(f"First value: {columns[0]}")

    print()


# ============================================================
# 3. RUN AUDIT FOR ALL VINTAGES
# ============================================================

def main():

    zip_files = sorted(RAW_DIR.glob("sample_*.zip"))

    print("\nMORTGAGE DATA INGESTION AUDIT")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"ZIP files found: {len(zip_files)}\n")

    if not zip_files:
        raise FileNotFoundError(
            f"No Freddie Mac ZIP files found in {RAW_DIR}"
        )

    for zip_path in zip_files:
        audit_zip(zip_path)


if __name__ == "__main__":
    main()
