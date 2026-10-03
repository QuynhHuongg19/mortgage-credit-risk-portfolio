from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MORTGAGE CREDIT RISK PORTFOLIO
# STEP 16 - BUILD DASHBOARD SUMMARY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# 1. INPUT FILES
# ============================================================

FILES = {
    "pd": OUTPUT_DIR / "competing_risks_pd_horizons.csv",
    "vintage": OUTPUT_DIR / "vintage_competing_risks_pd.csv",
    "cox": OUTPUT_DIR / "cox_default_risk_drivers.csv",
    "cox_tv": OUTPUT_DIR / "cox_time_varying_hr_by_month.csv",
    "fine_gray": OUTPUT_DIR / "fine_gray_default_risk_drivers.csv",
    "shock_exposure": OUTPUT_DIR / "shock_exposure_summary.csv",
    "shock_period": OUTPUT_DIR / "shock_period_analysis.csv",
}


# ============================================================
# 2. OUTPUT FILES
# ============================================================

SUMMARY_FILE = OUTPUT_DIR / "dashboard_summary.csv"
VINTAGE_FILE = OUTPUT_DIR / "dashboard_vintage_pd.csv"
RISK_DRIVER_FILE = OUTPUT_DIR / "dashboard_risk_drivers.csv"
SHOCK_FILE = OUTPUT_DIR / "dashboard_shock_analysis.csv"


# ============================================================
# 3. HELPERS
# ============================================================

def require_files():
    missing = [
        str(path)
        for path in FILES.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required output files:\n"
            + "\n".join(missing)
        )


def read(name):
    return pd.read_csv(
        FILES[name],
        low_memory=False,
    )


def find_column(df, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate = candidate.strip().lower()

        if candidate in lookup:
            return lookup[candidate]

    return None


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# 4. OVERALL COMPETING-RISK PD
# ============================================================

def build_pd_summary(pd_df):

    horizon_col = find_column(
        pd_df,
        [
            "horizon_months",
            "horizon",
            "month",
        ],
    )

    default_col = find_column(
        pd_df,
        [
            "pd_default",
            "default_cif",
            "default_pd",
            "pd",
        ],
    )

    prepay_col = find_column(
        pd_df,
        [
            "cif_prepayment",
            "prepayment_cif",
            "prepay_cif",
            "prepayment_pd",
        ],
    )

    event_free_col = find_column(
        pd_df,
        [
            "event_free_probability",
            "event_free",
            "survival_probability",
        ],
    )

    required = {
        "horizon": horizon_col,
        "default CIF": default_col,
        "prepayment CIF": prepay_col,
    }

    missing = [
        name
        for name, column in required.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            "PD horizon file missing columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + str(list(pd_df.columns))
        )

    rows = []

    for horizon in [12, 24, 36, 60]:

        subset = pd_df[
            numeric(pd_df[horizon_col]).eq(horizon)
        ]

        if subset.empty:
            raise ValueError(
                f"Missing competing-risk estimate "
                f"for horizon {horizon} months."
            )

        row = subset.iloc[0]

        default_cif = float(
            row[default_col]
        )

        prepayment_cif = float(
            row[prepay_col]
        )

        if event_free_col is not None:
            event_free = float(
                row[event_free_col]
            )
        else:
            event_free = (
                1.0
                - default_cif
                - prepayment_cif
            )

        rows.append(
            {
                "section": "OVERALL_PD",
                "metric": f"PD{horizon}",
                "horizon_months": horizon,
                "default_cif": default_cif,
                "prepayment_cif": prepayment_cif,
                "event_free_probability": event_free,
                "analysis_role": (
                    "SENSITIVITY"
                    if horizon == 60
                    else "PRIMARY"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 5. VINTAGE PD
# ============================================================

def build_vintage_summary(vintage_df):

    vintage_col = find_column(
        vintage_df,
        ["vintage"],
    )

    if vintage_col is None:
        raise ValueError(
            "Vintage output does not contain "
            "a vintage column.\n"
            f"Available columns: {list(vintage_df.columns)}"
        )

    result = pd.DataFrame(
        {
            "vintage": numeric(
                vintage_df[vintage_col]
            )
        }
    )

    # --------------------------------------------------------
    # Support wide-format output
    # --------------------------------------------------------

    wide_pd12 = find_column(
        vintage_df,
        [
            "pd12",
            "pd_12",
            "default_cif_12",
        ],
    )

    wide_pd24 = find_column(
        vintage_df,
        [
            "pd24",
            "pd_24",
            "default_cif_24",
        ],
    )

    wide_pd36 = find_column(
        vintage_df,
        [
            "pd36",
            "pd_36",
            "default_cif_36",
        ],
    )

    if all(
        column is not None
        for column in [
            wide_pd12,
            wide_pd24,
            wide_pd36,
        ]
    ):

        result["pd12"] = numeric(
            vintage_df[wide_pd12]
        )

        result["pd24"] = numeric(
            vintage_df[wide_pd24]
        )

        result["pd36"] = numeric(
            vintage_df[wide_pd36]
        )

    else:

        # ----------------------------------------------------
        # Support long-format output
        # ----------------------------------------------------

        horizon_col = find_column(
            vintage_df,
            [
                "horizon_months",
                "horizon",
                "month",
            ],
        )

        default_col = find_column(
            vintage_df,
            [
                "pd_default",
                "default_cif",
                "default_pd",
                "pd",
            ],
        )

        if (
            horizon_col is None
            or default_col is None
        ):
            raise ValueError(
                "Could not identify vintage PD columns.\n"
                f"Available columns: {list(vintage_df.columns)}"
            )

        temp = vintage_df[
            [
                vintage_col,
                horizon_col,
                default_col,
            ]
        ].copy()

        temp[horizon_col] = numeric(
            temp[horizon_col]
        )

        temp[default_col] = numeric(
            temp[default_col]
        )

        temp = temp[
            temp[horizon_col].isin(
                [12, 24, 36]
            )
        ]

        pivot = temp.pivot_table(
            index=vintage_col,
            columns=horizon_col,
            values=default_col,
            aggfunc="first",
        )

        pivot = pivot.rename(
            columns={
                12: "pd12",
                24: "pd24",
                36: "pd36",
            }
        ).reset_index()

        pivot = pivot.rename(
            columns={
                vintage_col: "vintage"
            }
        )

        result = pivot.copy()

    # --------------------------------------------------------
    # 36-month maturity status
    #
    # Project reporting convention:
    # >=95% calendar eligibility = primary vintage comparison.
    #
    # Based on Step 08:
    # 2016-2022 are mature/comparable at 36 months.
    # 2023-2026 are maturity-limited.
    # --------------------------------------------------------

    result["pd36_status"] = np.where(
        result["vintage"].between(
            2016,
            2022,
            inclusive="both",
        ),
        "FULLY_COMPARABLE",
        "MATURITY_LIMITED",
    )

    result["pd36_primary_comparison"] = (
        result["pd36_status"]
        .eq("FULLY_COMPARABLE")
    )

    result["maturity_note"] = np.where(
        result["pd36_primary_comparison"],
        (
            "Included in primary 36-month "
            "vintage comparison."
        ),
        (
            "36-month estimate is maturity-limited "
            "and should not be interpreted as fully "
            "comparable with mature vintages."
        ),
    )

    return (
        result
        .sort_values("vintage")
        .reset_index(drop=True)
    )


# ============================================================
# 6. CAUSE-SPECIFIC COX + FINE-GRAY
# ============================================================

def build_risk_driver_summary(
    cox_df,
    tv_df,
    fg_df,
):

    cox_var = find_column(
        cox_df,
        ["variable"],
    )

    cox_hr = find_column(
        cox_df,
        [
            "hazard_ratio",
            "hr",
        ],
    )

    cox_p = find_column(
        cox_df,
        [
            "p_value",
            "p",
        ],
    )

    fg_var = find_column(
        fg_df,
        ["variable"],
    )

    fg_shr = find_column(
        fg_df,
        [
            "subdistribution_hazard_ratio",
            "shr",
        ],
    )

    if (
        cox_var is None
        or cox_hr is None
        or fg_var is None
        or fg_shr is None
    ):
        raise ValueError(
            "Could not identify Cox/Fine-Gray columns.\n"
            f"Cox columns: {list(cox_df.columns)}\n"
            f"Fine-Gray columns: {list(fg_df.columns)}"
        )

    result = pd.DataFrame(
        {
            "variable": (
                cox_df[cox_var]
                .astype(str)
            ),
            "cause_specific_hr": numeric(
                cox_df[cox_hr]
            ),
        }
    )

    if cox_p is not None:
        result["cox_p_value"] = numeric(
            cox_df[cox_p]
        )
    else:
        result["cox_p_value"] = np.nan

    fg = pd.DataFrame(
        {
            "variable": (
                fg_df[fg_var]
                .astype(str)
            ),
            "fine_gray_shr": numeric(
                fg_df[fg_shr]
            ),
        }
    )

    result = result.merge(
        fg,
        on="variable",
        how="left",
        validate="one_to_one",
    )

    result[
        "fine_gray_inference_available"
    ] = False

    result[
        "fine_gray_note"
    ] = (
        "Point estimate only; variance=False "
        "was used for full-cohort Fine-Gray estimation."
    )

    # --------------------------------------------------------
    # Time-varying Cox output
    # --------------------------------------------------------

    tv_var = find_column(
        tv_df,
        ["variable"],
    )

    tv_month = find_column(
        tv_df,
        [
            "month",
            "evaluation_month",
            "horizon_months",
        ],
    )

    tv_hr = find_column(
        tv_df,
        [
            "hazard_ratio",
            "hr",
            "time_dependent_hr",
        ],
    )

    if (
        tv_var is not None
        and tv_month is not None
        and tv_hr is not None
    ):

        for month in [12, 24, 36]:

            temp = tv_df[
                numeric(
                    tv_df[tv_month]
                ).eq(month)
            ][
                [
                    tv_var,
                    tv_hr,
                ]
            ].copy()

            temp.columns = [
                "variable",
                f"time_varying_hr_{month}m",
            ]

            temp["variable"] = (
                temp["variable"]
                .astype(str)
            )

            temp[
                f"time_varying_hr_{month}m"
            ] = numeric(
                temp[
                    f"time_varying_hr_{month}m"
                ]
            )

            temp = temp.drop_duplicates(
                subset=["variable"]
            )

            result = result.merge(
                temp,
                on="variable",
                how="left",
                validate="one_to_one",
            )

    else:

        result["time_varying_hr_12m"] = np.nan
        result["time_varying_hr_24m"] = np.nan
        result["time_varying_hr_36m"] = np.nan

    # Step 14 is retained as an exploratory PH diagnostic.
    # It must not be presented as the final time-varying model
    # until a valid start-stop/risk-set implementation is used.

    result[
        "time_varying_model_status"
    ] = "EXPLORATORY_NOT_FINAL"

    result[
        "time_varying_model_note"
    ] = (
        "Exploratory time-interaction results. "
        "Do not use as final inference until "
        "validated with a proper start-stop "
        "time-varying Cox implementation."
    )

    return result


# ============================================================
# 7. MACROECONOMIC SHOCK TABLE
# ============================================================

def build_shock_summary(
    exposure_df,
    period_df,
):

    exposure = exposure_df.copy()

    exposure.insert(
        0,
        "analysis_type",
        "FIRST_36_MONTH_EXPOSURE",
    )

    period = period_df.copy()

    period.insert(
        0,
        "analysis_type",
        "BEFORE_DURING_AFTER",
    )

    columns = list(
        dict.fromkeys(
            list(exposure.columns)
            + list(period.columns)
        )
    )

    exposure = exposure.reindex(
        columns=columns
    )

    period = period.reindex(
        columns=columns
    )

    result = pd.concat(
        [
            exposure,
            period,
        ],
        ignore_index=True,
    )

    result[
        "causal_interpretation_allowed"
    ] = False

    return result


# ============================================================
# 8. VALIDATION
# ============================================================

def validate_pd(pd_summary):

    errors = []

    if set(
        pd_summary["horizon_months"]
    ) != {12, 24, 36, 60}:
        errors.append(
            "PD horizons must be exactly "
            "12, 24, 36 and 60 months."
        )

    for column in [
        "default_cif",
        "prepayment_cif",
        "event_free_probability",
    ]:

        values = numeric(
            pd_summary[column]
        )

        if values.isna().any():
            errors.append(
                f"Missing probability in {column}."
            )

        invalid = (
            (values < -1e-8)
            | (values > 1 + 1e-8)
        )

        if invalid.any():
            errors.append(
                f"Probability outside [0,1] "
                f"in {column}."
            )

    probability_sum = (
        pd_summary["default_cif"]
        + pd_summary["prepayment_cif"]
        + pd_summary[
            "event_free_probability"
        ]
    )

    if not np.allclose(
        probability_sum,
        1.0,
        atol=1e-5,
    ):
        errors.append(
            "Default CIF + Prepayment CIF + "
            "event-free probability does not sum "
            "to approximately 1."
        )

    return errors


def validate_vintage(vintage_summary):

    errors = []

    if vintage_summary[
        "vintage"
    ].duplicated().any():
        errors.append(
            "Duplicate vintage rows detected."
        )

    for column in [
        "pd12",
        "pd24",
        "pd36",
    ]:

        if column not in vintage_summary.columns:
            errors.append(
                f"Missing {column}."
            )

            continue

        values = numeric(
            vintage_summary[column]
        )

        invalid = (
            (values < -1e-8)
            | (values > 1 + 1e-8)
        )

        if invalid.any():
            errors.append(
                f"Invalid probability in {column}."
            )

    return errors


def validate_risk(risk_summary):

    errors = []

    if risk_summary[
        "variable"
    ].duplicated().any():
        errors.append(
            "Duplicate risk-driver rows detected."
        )

    for column in [
        "cause_specific_hr",
        "fine_gray_shr",
    ]:

        values = numeric(
            risk_summary[column]
        )

        if values.isna().any():
            errors.append(
                f"Missing estimates in {column}."
            )

        if (values <= 0).any():
            errors.append(
                f"Non-positive estimates in {column}."
            )

    return errors


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print()
    print("MORTGAGE CREDIT RISK")
    print(
        "STEP 16 - BUILD DASHBOARD SUMMARY"
    )
    print()

    require_files()

    pd_df = read("pd")
    vintage_df = read("vintage")
    cox_df = read("cox")
    tv_df = read("cox_tv")
    fg_df = read("fine_gray")
    exposure_df = read("shock_exposure")
    period_df = read("shock_period")

    # --------------------------------------------------------
    # Build dashboard datasets
    # --------------------------------------------------------

    pd_summary = build_pd_summary(
        pd_df
    )

    vintage_summary = (
        build_vintage_summary(
            vintage_df
        )
    )

    risk_summary = (
        build_risk_driver_summary(
            cox_df,
            tv_df,
            fg_df,
        )
    )

    shock_summary = (
        build_shock_summary(
            exposure_df,
            period_df,
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    errors = []

    errors.extend(
        validate_pd(
            pd_summary
        )
    )

    errors.extend(
        validate_vintage(
            vintage_summary
        )
    )

    errors.extend(
        validate_risk(
            risk_summary
        )
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "OVERALL COMPETING-RISK PD"
    )
    print("=" * 72)

    for _, row in pd_summary.iterrows():

        horizon = int(
            row["horizon_months"]
        )

        print(
            f"PD{horizon:<2} "
            f"= {row['default_cif']:.4%} | "
            f"Prepay CIF = "
            f"{row['prepayment_cif']:.4%} | "
            f"{row['analysis_role']}"
        )

    print()
    print("=" * 72)
    print(
        "36-MONTH VINTAGE COMPARABILITY"
    )
    print("=" * 72)

    comparable = (
        vintage_summary.loc[
            vintage_summary[
                "pd36_primary_comparison"
            ],
            "vintage",
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    limited = (
        vintage_summary.loc[
            ~vintage_summary[
                "pd36_primary_comparison"
            ],
            "vintage",
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    print(
        "Primary comparison vintages : "
        + ", ".join(
            map(str, comparable)
        )
    )

    print(
        "Maturity-limited vintages   : "
        + ", ".join(
            map(str, limited)
        )
    )

    print()
    print("=" * 72)
    print("RISK DRIVER MODELS")
    print("=" * 72)

    display_columns = [
        "variable",
        "cause_specific_hr",
        "fine_gray_shr",
    ]

    for column in [
        "time_varying_hr_12m",
        "time_varying_hr_24m",
        "time_varying_hr_36m",
    ]:
        if column in risk_summary.columns:
            display_columns.append(
                column
            )

    print(
        risk_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print(
        f"Validation errors: {len(errors)}"
    )

    if errors:

        for error in errors:
            print(
                f"  - {error}"
            )

        raise ValueError(
            "Step 16 validation failed."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    pd_summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    vintage_summary.to_csv(
        VINTAGE_FILE,
        index=False,
    )

    risk_summary.to_csv(
        RISK_DRIVER_FILE,
        index=False,
    )

    shock_summary.to_csv(
        SHOCK_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("DASHBOARD OUTPUTS")
    print("=" * 72)

    print(
        f"Overall PD   : {SUMMARY_FILE.name}"
    )

    print(
        f"Vintage PD   : {VINTAGE_FILE.name}"
    )

    print(
        f"Risk drivers : {RISK_DRIVER_FILE.name}"
    )

    print(
        f"Shock analysis: {SHOCK_FILE.name}"
    )

    print()
    print(
        "IMPORTANT MODELING NOTES"
    )

    print(
        "- PD12/PD24/PD36 use "
        "Aalen-Johansen competing-risk CIF."
    )

    print(
        "- PD60 is retained as "
        "sensitivity analysis."
    )

    print(
        "- 2016-2022 are used for the "
        "primary mature PD36 vintage comparison."
    )

    print(
        "- 2023-2026 PD36 values are "
        "maturity-limited."
    )

    print(
        "- Fine-Gray estimates are "
        "full-cohort point estimates only."
    )

    print(
        "- Step 14 time-varying estimates "
        "remain exploratory and are not final inference."
    )

    print(
        "- Shock results are descriptive "
        "associations, not causal effects."
    )

    print()
    print("STEP 16 COMPLETE")


if __name__ == "__main__":
    main()