from pathlib import Path
from collections import defaultdict, Counter
import csv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "survival_loan_level.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "followup_horizon_audit.csv"
)

HORIZONS = [12, 24, 36, 60]


# ============================================================
# 2. SAFE CONVERSION
# ============================================================

def to_int(value):

    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


# ============================================================
# 3. STORAGE
# ============================================================

def new_vintage_stats():

    return {
        "total": 0,

        "event_counts": Counter(),

        "missing_duration": 0,
        "negative_duration": 0,

        "min_duration": None,
        "max_duration": None,

        "horizons": {
            h: {
                "reached": 0,
                "known": 0,

                "default_by": 0,
                "prepay_by": 0,

                "censored_before": 0,
                "event_free_at": 0
            }
            for h in HORIZONS
        }
    }


# ============================================================
# 4. UPDATE MIN / MAX
# ============================================================

def update_duration_range(
    stats,
    duration
):

    if stats["min_duration"] is None:
        stats["min_duration"] = duration
    else:
        stats["min_duration"] = min(
            stats["min_duration"],
            duration
        )

    if stats["max_duration"] is None:
        stats["max_duration"] = duration
    else:
        stats["max_duration"] = max(
            stats["max_duration"],
            duration
        )


# ============================================================
# 5. READ AND AUDIT
# ============================================================

def audit():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_FILE}"
        )

    stats_by_vintage = defaultdict(
        new_vintage_stats
    )

    overall = new_vintage_stats()

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "loan_id",
            "vintage",
            "duration_months",
            "event_code",
            "event_type",
            "event_loan_age",
            "last_observed_loan_age"
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row in reader:

            vintage = row[
                "vintage"
            ].strip()

            duration = to_int(
                row["duration_months"]
            )

            event_code = to_int(
                row["event_code"]
            )

            event_type = row[
                "event_type"
            ].strip()

            vintage_stats = (
                stats_by_vintage[vintage]
            )

            for stats in (
                vintage_stats,
                overall
            ):

                stats["total"] += 1

                stats[
                    "event_counts"
                ][event_code] += 1

                if duration is None:

                    stats[
                        "missing_duration"
                    ] += 1

                    continue

                if duration < 0:

                    stats[
                        "negative_duration"
                    ] += 1

                update_duration_range(
                    stats,
                    duration
                )

                # ============================================
                # HORIZON AUDIT
                # ============================================

                for horizon in HORIZONS:

                    hstats = stats[
                        "horizons"
                    ][horizon]

                    # ----------------------------------------
                    # Reached horizon:
                    # actual follow-up/event time >= horizon.
                    # ----------------------------------------

                    if duration >= horizon:

                        hstats[
                            "reached"
                        ] += 1

                    # ----------------------------------------
                    # Default known by horizon
                    # ----------------------------------------

                    if (
                        event_code == 1
                        and duration <= horizon
                    ):

                        hstats[
                            "default_by"
                        ] += 1

                        hstats[
                            "known"
                        ] += 1

                    # ----------------------------------------
                    # Prepayment known by horizon
                    # ----------------------------------------

                    elif (
                        event_code == 2
                        and duration <= horizon
                    ):

                        hstats[
                            "prepay_by"
                        ] += 1

                        hstats[
                            "known"
                        ] += 1

                    # ----------------------------------------
                    # Censored before horizon:
                    # horizon outcome is not fully observed.
                    # ----------------------------------------

                    elif (
                        event_code == 0
                        and duration < horizon
                    ):

                        hstats[
                            "censored_before"
                        ] += 1

                    # ----------------------------------------
                    # Still observed/event-free at horizon
                    # ----------------------------------------

                    elif duration >= horizon:

                        hstats[
                            "event_free_at"
                        ] += 1

                        hstats[
                            "known"
                        ] += 1

                    # ----------------------------------------
                    # Defensive case
                    # ----------------------------------------

                    else:

                        raise ValueError(
                            "Unexpected horizon "
                            "classification for "
                            f"loan {row['loan_id']} "
                            f"at {horizon} months."
                        )

    return (
        stats_by_vintage,
        overall
    )


# ============================================================
# 6. CREATE OUTPUT ROWS
# ============================================================

def build_output_rows(
    stats_by_vintage
):

    rows = []

    for vintage in sorted(
        stats_by_vintage,
        key=int
    ):

        stats = stats_by_vintage[
            vintage
        ]

        total = stats["total"]

        for horizon in HORIZONS:

            hstats = stats[
                "horizons"
            ][horizon]

            known = hstats[
                "known"
            ]

            censored_before = hstats[
                "censored_before"
            ]

            # Accounting check:
            # Every loan must be either known by horizon
            # or censored before horizon.

            accounting_ok = (
                known
                + censored_before
                == total
            )

            rows.append(
                {
                    "vintage":
                        vintage,

                    "horizon_months":
                        horizon,

                    "total_loans":
                        total,

                    "reached_horizon":
                        hstats[
                            "reached"
                        ],

                    "known_by_horizon":
                        known,

                    "known_pct":
                        round(
                            100
                            * known
                            / total,
                            2
                        )
                        if total else 0,

                    "default_by_horizon":
                        hstats[
                            "default_by"
                        ],

                    "prepay_by_horizon":
                        hstats[
                            "prepay_by"
                        ],

                    "event_free_at_horizon":
                        hstats[
                            "event_free_at"
                        ],

                    "censored_before_horizon":
                        censored_before,

                    "accounting_ok":
                        accounting_ok
                }
            )

    return rows


# ============================================================
# 7. SAVE CSV
# ============================================================

def save_output(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "vintage",
        "horizon_months",
        "total_loans",
        "reached_horizon",
        "known_by_horizon",
        "known_pct",
        "default_by_horizon",
        "prepay_by_horizon",
        "event_free_at_horizon",
        "censored_before_horizon",
        "accounting_ok"
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
        writer.writerows(rows)


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

def print_summary(
    stats_by_vintage,
    overall
):

    print(
        "\nMORTGAGE CREDIT RISK - "
        "FOLLOW-UP HORIZON AUDIT\n"
    )

    print("=" * 92)

    print(
        f"{'Vintage':<9}"
        f"{'Loans':>9}"
        f"{'MaxAge':>9}"
        f"{'12m %':>10}"
        f"{'24m %':>10}"
        f"{'36m %':>10}"
        f"{'60m %':>10}"
    )

    print("=" * 92)

    for vintage in sorted(
        stats_by_vintage,
        key=int
    ):

        stats = stats_by_vintage[
            vintage
        ]

        total = stats["total"]

        percentages = []

        for horizon in HORIZONS:

            known = stats[
                "horizons"
            ][horizon]["known"]

            pct = (
                100 * known / total
                if total
                else 0
            )

            percentages.append(pct)

        print(
            f"{vintage:<9}"
            f"{total:>9,}"
            f"{str(stats['max_duration']):>9}"
            f"{percentages[0]:>9.2f}%"
            f"{percentages[1]:>9.2f}%"
            f"{percentages[2]:>9.2f}%"
            f"{percentages[3]:>9.2f}%"
        )

    print("\n" + "=" * 92)
    print("OVERALL FOLLOW-UP AUDIT")
    print("=" * 92)

    print(
        f"Total loans       : "
        f"{overall['total']:,}"
    )

    print(
        f"Minimum duration  : "
        f"{overall['min_duration']}"
    )

    print(
        f"Maximum duration  : "
        f"{overall['max_duration']}"
    )

    print(
        f"Missing duration  : "
        f"{overall['missing_duration']:,}"
    )

    print(
        f"Negative duration : "
        f"{overall['negative_duration']:,}"
    )

    for horizon in HORIZONS:

        hstats = overall[
            "horizons"
        ][horizon]

        known = hstats["known"]

        pct = (
            100
            * known
            / overall["total"]
        )

        print(
            f"\nHorizon {horizon:>2} months"
        )

        print(
            f"  Known outcome       : "
            f"{known:,} "
            f"({pct:.2f}%)"
        )

        print(
            f"  Default by horizon  : "
            f"{hstats['default_by']:,}"
        )

        print(
            f"  Prepay by horizon   : "
            f"{hstats['prepay_by']:,}"
        )

        print(
            f"  Event-free at H     : "
            f"{hstats['event_free_at']:,}"
        )

        print(
            f"  Censored before H   : "
            f"{hstats['censored_before']:,}"
        )

        print(
            f"  Accounting check    : "
            f"{known + hstats['censored_before'] == overall['total']}"
        )

    print(
        "\nEvent totals:"
    )

    print(
        f"  Censor     : "
        f"{overall['event_counts'][0]:,}"
    )

    print(
        f"  Default    : "
        f"{overall['event_counts'][1]:,}"
    )

    print(
        f"  Prepayment : "
        f"{overall['event_counts'][2]:,}"
    )

    print(
        f"\nAudit saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nFollow-up horizon audit complete."
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    (
        stats_by_vintage,
        overall
    ) = audit()

    rows = build_output_rows(
        stats_by_vintage
    )

    save_output(
        rows
    )

    print_summary(
        stats_by_vintage,
        overall
    )


if __name__ == "__main__":
    main()