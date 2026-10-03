from pathlib import Path
from zipfile import ZipFile


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# RAW PERFORMANCE PERIOD AUDIT
# ============================================================

def main():

    zip_files = sorted(
        RAW_DIR.glob("sample_*.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            f"No sample ZIP files found in:\n{RAW_DIR}"
        )

    overall_min = None
    overall_max = None
    overall_rows = 0

    print(
        "\nMORTGAGE CREDIT RISK - "
        "RAW PERFORMANCE PERIOD AUDIT\n"
    )

    print("=" * 70)

    for zip_path in zip_files:

        vintage = zip_path.stem.split("_")[-1]

        performance_name = (
            f"sample_perf_{vintage}.txt"
        )

        vintage_min = None
        vintage_max = None
        vintage_rows = 0

        with ZipFile(zip_path, "r") as archive:

            if performance_name not in archive.namelist():
                raise FileNotFoundError(
                    f"{performance_name} not found "
                    f"in {zip_path.name}"
                )

            with archive.open(
                performance_name
            ) as file:

                for raw_line in file:

                    line = raw_line.decode(
                        "utf-8",
                        errors="replace"
                    )

                    fields = line.rstrip(
                        "\r\n"
                    ).split("|")

                    if len(fields) < 2:
                        continue

                    period = fields[1].strip()

                    if (
                        len(period) != 6
                        or not period.isdigit()
                    ):
                        continue

                    vintage_rows += 1
                    overall_rows += 1

                    if (
                        vintage_min is None
                        or period < vintage_min
                    ):
                        vintage_min = period

                    if (
                        vintage_max is None
                        or period > vintage_max
                    ):
                        vintage_max = period

                    if (
                        overall_min is None
                        or period < overall_min
                    ):
                        overall_min = period

                    if (
                        overall_max is None
                        or period > overall_max
                    ):
                        overall_max = period

        print(
            f"{vintage} | "
            f"Rows: {vintage_rows:>10,} | "
            f"Min: {vintage_min} | "
            f"Max: {vintage_max}"
        )

    print("=" * 70)

    print("\nOVERALL RAW PERFORMANCE PERIOD AUDIT")
    print("=" * 70)

    print(
        f"Performance rows : "
        f"{overall_rows:,}"
    )

    print(
        f"MIN PERIOD       : "
        f"{overall_min}"
    )

    print(
        f"MAX PERIOD       : "
        f"{overall_max}"
    )

    print(
        "\nRaw performance period audit complete."
    )


if __name__ == "__main__":
    main()