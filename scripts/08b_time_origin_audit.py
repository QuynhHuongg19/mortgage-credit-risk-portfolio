from pathlib import Path
from collections import Counter, defaultdict
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
    / "time_origin_audit.csv"
)

DATA_CUTOFF = "202603"


# ============================================================
# 2. HELPERS
# ============================================================

def clean_value(value):

    if value is None:
        return ""

    value = value.strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def parse_yyyymm(value):

    value = clean_value(value)

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

    return year, month


def month_index(year, month):

    return year * 12 + month - 1


def months_between(start, end):

    return (
        month_index(*end)
        - month_index(*start)
    )


def parse_int(value):

    value = clean_value(value)

    if value == "":
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


# ============================================================
# 3. AUDIT
# ============================================================

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_FILE}"
        )

    cutoff = parse_yyyymm(
        DATA_CUTOFF
    )

    if cutoff is None:
        raise ValueError(
            "Invalid DATA_CUTOFF."
        )

    total = 0

    invalid_first_payment = 0
    first_payment_after_cutoff = 0

    invalid_first_period = 0
    invalid_last_period = 0

    invalid_first_age = 0
    invalid_last_age = 0

    first_age_distribution = Counter()

    start_alignment_distribution = Counter()
    followup_alignment_distribution = Counter()

    by_vintage = defaultdict(
        lambda: {
            "total": 0,
            "fp_after_cutoff": 0,
            "first_age_0": 0,
            "first_age_1": 0,
            "first_age_other": 0,
            "start_match": 0,
            "start_mismatch": 0,
            "followup_match": 0,
            "followup_mismatch": 0,
        }
    )

    examples_fp_after_cutoff = []
    examples_start_mismatch = []
    examples_followup_mismatch = []

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "loan_id",
            "vintage",
            "first_payment_date",
            "first_observed_period",
            "last_observed_period",
            "first_observed_loan_age",
            "last_observed_loan_age",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        for row in reader:

            total += 1

            loan_id = clean_value(
                row["loan_id"]
            )

            vintage = clean_value(
                row["vintage"]
            )

            stats = by_vintage[vintage]
            stats["total"] += 1

            first_payment = parse_yyyymm(
                row["first_payment_date"]
            )

            first_period = parse_yyyymm(
                row["first_observed_period"]
            )

            last_period = parse_yyyymm(
                row["last_observed_period"]
            )

            first_age = parse_int(
                row["first_observed_loan_age"]
            )

            last_age = parse_int(
                row["last_observed_loan_age"]
            )

            # --------------------------------------------
            # First payment date vs raw data cutoff
            # --------------------------------------------

            if first_payment is None:

                invalid_first_payment += 1

            elif first_payment > cutoff:

                first_payment_after_cutoff += 1
                stats["fp_after_cutoff"] += 1

                if (
                    len(
                        examples_fp_after_cutoff
                    ) < 20
                ):
                    examples_fp_after_cutoff.append(
                        {
                            "loan_id": loan_id,
                            "vintage": vintage,
                            "first_payment_date":
                                clean_value(
                                    row[
                                        "first_payment_date"
                                    ]
                                ),
                            "first_observed_period":
                                clean_value(
                                    row[
                                        "first_observed_period"
                                    ]
                                ),
                            "first_observed_loan_age":
                                clean_value(
                                    row[
                                        "first_observed_loan_age"
                                    ]
                                ),
                        }
                    )

            # --------------------------------------------
            # First observed loan age
            # --------------------------------------------

            if first_age is None:

                invalid_first_age += 1

            else:

                first_age_distribution[
                    first_age
                ] += 1

                if first_age == 0:
                    stats["first_age_0"] += 1

                elif first_age == 1:
                    stats["first_age_1"] += 1

                else:
                    stats[
                        "first_age_other"
                    ] += 1

            # --------------------------------------------
            # Validate first observed period
            # --------------------------------------------

            if first_period is None:
                invalid_first_period += 1

            # --------------------------------------------
            # Validate last observed values
            # --------------------------------------------

            if last_period is None:
                invalid_last_period += 1

            if last_age is None:
                invalid_last_age += 1

            # --------------------------------------------
            # START ALIGNMENT
            #
            # Compare:
            # calendar months from first payment date
            # to first observed performance period
            #
            # versus first observed loan age.
            #
            # We retain the raw difference rather than
            # assuming in advance whether Freddie Mac's
            # convention is age 0 or age 1 at first payment.
            # --------------------------------------------

            if (
                first_payment is not None
                and first_period is not None
                and first_age is not None
            ):

                calendar_start_gap = (
                    months_between(
                        first_payment,
                        first_period
                    )
                )

                start_difference = (
                    first_age
                    - calendar_start_gap
                )

                start_alignment_distribution[
                    start_difference
                ] += 1

                if start_difference in (0, 1):

                    stats[
                        "start_match"
                    ] += 1

                else:

                    stats[
                        "start_mismatch"
                    ] += 1

                    if (
                        len(
                            examples_start_mismatch
                        ) < 20
                    ):

                        examples_start_mismatch.append(
                            {
                                "loan_id": loan_id,
                                "vintage": vintage,
                                "first_payment_date":
                                    clean_value(
                                        row[
                                            "first_payment_date"
                                        ]
                                    ),
                                "first_observed_period":
                                    clean_value(
                                        row[
                                            "first_observed_period"
                                        ]
                                    ),
                                "first_observed_loan_age":
                                    first_age,
                                "calendar_start_gap":
                                    calendar_start_gap,
                                "difference":
                                    start_difference,
                            }
                        )

            # --------------------------------------------
            # WITHIN-PERFORMANCE ALIGNMENT
            #
            # This is the most important consistency check:
            #
            # last loan age - first loan age
            #
            # should equal
            #
            # calendar months between first and last
            # observed performance period.
            # --------------------------------------------

            if (
                first_period is not None
                and last_period is not None
                and first_age is not None
                and last_age is not None
            ):

                calendar_followup = (
                    months_between(
                        first_period,
                        last_period
                    )
                )

                loan_age_followup = (
                    last_age
                    - first_age
                )

                followup_difference = (
                    loan_age_followup
                    - calendar_followup
                )

                followup_alignment_distribution[
                    followup_difference
                ] += 1

                if followup_difference == 0:

                    stats[
                        "followup_match"
                    ] += 1

                else:

                    stats[
                        "followup_mismatch"
                    ] += 1

                    if (
                        len(
                            examples_followup_mismatch
                        ) < 20
                    ):

                        examples_followup_mismatch.append(
                            {
                                "loan_id": loan_id,
                                "vintage": vintage,
                                "first_observed_period":
                                    clean_value(
                                        row[
                                            "first_observed_period"
                                        ]
                                    ),
                                "last_observed_period":
                                    clean_value(
                                        row[
                                            "last_observed_period"
                                        ]
                                    ),
                                "first_observed_loan_age":
                                    first_age,
                                "last_observed_loan_age":
                                    last_age,
                                "calendar_followup":
                                    calendar_followup,
                                "loan_age_followup":
                                    loan_age_followup,
                                "difference":
                                    followup_difference,
                            }
                        )

    # ========================================================
    # 4. SAVE VINTAGE SUMMARY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "vintage",
        "total_loans",
        "first_payment_after_cutoff",
        "first_age_0",
        "first_age_1",
        "first_age_other",
        "start_match_0_or_1",
        "start_mismatch",
        "followup_exact_match",
        "followup_mismatch",
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

        for vintage in sorted(
            by_vintage,
            key=int
        ):

            s = by_vintage[vintage]

            writer.writerow(
                {
                    "vintage":
                        vintage,

                    "total_loans":
                        s["total"],

                    "first_payment_after_cutoff":
                        s[
                            "fp_after_cutoff"
                        ],

                    "first_age_0":
                        s["first_age_0"],

                    "first_age_1":
                        s["first_age_1"],

                    "first_age_other":
                        s[
                            "first_age_other"
                        ],

                    "start_match_0_or_1":
                        s["start_match"],

                    "start_mismatch":
                        s[
                            "start_mismatch"
                        ],

                    "followup_exact_match":
                        s[
                            "followup_match"
                        ],

                    "followup_mismatch":
                        s[
                            "followup_mismatch"
                        ],
                }
            )

    # ========================================================
    # 5. PRINT RESULTS
    # ========================================================

    print(
        "\nMORTGAGE CREDIT RISK - "
        "TIME ORIGIN AUDIT\n"
    )

    print("=" * 78)

    print(
        f"Loans audited                  : "
        f"{total:,}"
    )

    print(
        f"Raw performance cutoff         : "
        f"{DATA_CUTOFF}"
    )

    print(
        f"First payment after cutoff     : "
        f"{first_payment_after_cutoff:,}"
    )

    print(
        f"Invalid first payment date     : "
        f"{invalid_first_payment:,}"
    )

    print(
        f"Invalid first observed period  : "
        f"{invalid_first_period:,}"
    )

    print(
        f"Invalid last observed period   : "
        f"{invalid_last_period:,}"
    )

    print(
        f"Invalid first observed age     : "
        f"{invalid_first_age:,}"
    )

    print(
        f"Invalid last observed age      : "
        f"{invalid_last_age:,}"
    )

    print("\n" + "=" * 78)

    print(
        "FIRST OBSERVED LOAN AGE DISTRIBUTION"
    )

    print("=" * 78)

    for age, count in sorted(
        first_age_distribution.items()
    ):

        print(
            f"Loan age {age:>3}: "
            f"{count:>10,}"
        )

    print("\n" + "=" * 78)

    print(
        "START ALIGNMENT DIFFERENCE"
    )

    print(
        "difference = first_observed_loan_age "
        "- calendar months from first payment "
        "to first observed period"
    )

    print("=" * 78)

    for difference, count in sorted(
        start_alignment_distribution.items()
    ):

        print(
            f"Difference {difference:>4}: "
            f"{count:>10,}"
        )

    print("\n" + "=" * 78)

    print(
        "WITHIN-PERFORMANCE FOLLOW-UP ALIGNMENT"
    )

    print(
        "difference = change in loan age "
        "- change in calendar month"
    )

    print("=" * 78)

    for difference, count in sorted(
        followup_alignment_distribution.items()
    ):

        print(
            f"Difference {difference:>4}: "
            f"{count:>10,}"
        )

    print("\n" + "=" * 78)

    print(
        "VINTAGE SUMMARY"
    )

    print("=" * 78)

    print(
        f"{'Vintage':<9}"
        f"{'Loans':>9}"
        f"{'FP>Cut':>9}"
        f"{'Age0':>9}"
        f"{'Age1':>9}"
        f"{'Other':>9}"
        f"{'FU Match':>11}"
        f"{'FU Mis':>9}"
    )

    print("-" * 78)

    for vintage in sorted(
        by_vintage,
        key=int
    ):

        s = by_vintage[vintage]

        print(
            f"{vintage:<9}"
            f"{s['total']:>9,}"
            f"{s['fp_after_cutoff']:>9,}"
            f"{s['first_age_0']:>9,}"
            f"{s['first_age_1']:>9,}"
            f"{s['first_age_other']:>9,}"
            f"{s['followup_match']:>11,}"
            f"{s['followup_mismatch']:>9,}"
        )

    if examples_fp_after_cutoff:

        print("\n" + "=" * 78)

        print(
            "EXAMPLES: FIRST PAYMENT AFTER CUTOFF"
        )

        print("=" * 78)

        for example in (
            examples_fp_after_cutoff
        ):

            print(example)

    if examples_start_mismatch:

        print("\n" + "=" * 78)

        print(
            "EXAMPLES: START ALIGNMENT MISMATCH"
        )

        print("=" * 78)

        for example in (
            examples_start_mismatch
        ):

            print(example)

    if examples_followup_mismatch:

        print("\n" + "=" * 78)

        print(
            "EXAMPLES: FOLLOW-UP ALIGNMENT MISMATCH"
        )

        print("=" * 78)

        for example in (
            examples_followup_mismatch
        ):

            print(example)

    print(
        f"\nAudit saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nTime origin audit complete."
    )


if __name__ == "__main__":
    main()