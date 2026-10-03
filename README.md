# Mortgage Credit Risk Analytics

An end-to-end mortgage credit risk analytics project using Freddie Mac Single-Family Loan-Level sample data from 2016–2026.

This project models mortgage default as a **time-to-event problem** while explicitly accounting for **Voluntary Prepayment as a competing event**. The analytical framework combines survival analysis, competing-risks estimation, vintage analysis, credit risk-driver modeling, macroeconomic-period analysis, and an interactive Streamlit dashboard.

## Project Overview

A mortgage can leave the active portfolio through more than one terminal event. In particular, a loan that prepays can no longer subsequently default.

For this reason, the project models two mutually exclusive study events:

- **Default:** first observed numeric delinquency status of 3 or greater, corresponding to 90+ days past due.
- **Voluntary Prepayment:** Zero Balance Code 01 occurring before Default.

Other relevant termination events are treated as censoring.

The project addresses four analytical questions:

1. How does cumulative mortgage Default risk evolve over loan age?
2. How does estimated Default risk change when Voluntary Prepayment is explicitly treated as a competing event?
3. How are borrower and loan characteristics associated with Default risk?
4. How do mortgage outcomes differ across origination vintages and selected macroeconomic periods?

## Data

**Source:** Freddie Mac Single-Family Loan-Level Dataset sample files.

| Item | Value |
|---|---:|
| Origination vintages | 2016–2026 |
| Origination records audited | 512,500 |
| Monthly performance records audited | 20,097,384 |
| Final survival cohort | 512,490 loans |

The dataset contains two primary components:

- **Origination data:** borrower and loan characteristics measured at origination.
- **Performance data:** monthly loan-level performance histories.

Ten origination records have no corresponding performance history. These observations are retained in data-quality auditing but excluded from survival modeling.

Raw loan-level files are intentionally excluded from Git. Freddie Mac field-header and reference files used for schema validation are retained under `docs/references/`.

## Event Definition

### Default

Default is defined as the first observed month in which numeric delinquency status satisfies:

**Delinquency status >= 3 (90+ days past due).**

Non-numeric delinquency states such as `RA` and `XX` are not interpreted as numeric delinquency thresholds.

### Voluntary Prepayment

Voluntary Prepayment is identified by:

**Zero Balance Code = 01**

when it occurs before Default.

### Other Terminations

Zero Balance Codes 15, 16, and 96 are treated as censoring when they occur before a study event.

When Default and Prepayment are recorded in the same month, the project convention assigns the observation to Default.

## Survival Time Origin

The survival clock begins at each loan's **first observed monthly performance period**.

The duration is calculated as:

**Duration = month difference between the first observed performance month and the event/censoring month.**

Therefore:

- first observed month = month 0;
- next calendar month = month 1;
- twelve months later = month 12.

Freddie Mac Loan Age and First Payment Date are retained for audit and contextual validation rather than used as the primary survival clock.

This calendar-based definition provides a consistent elapsed-time measure while avoiding reliance on occasional inconsistencies observed in reported Loan Age sequences.
## Competing-Risks Framework

Voluntary Prepayment is treated as a **competing event** because a prepaid loan is no longer exposed to future Default.

The primary quantities of interest are the cumulative incidence functions:

**Default CIF**

CIF_D(t) = P(T <= t, J = Default)

**Prepayment CIF**

CIF_P(t) = P(T <= t, J = Prepayment)

The project uses the **Aalen–Johansen estimator** as the primary method for estimating cumulative Default probability in the presence of competing Prepayment.

## Default Risk by Horizon

| Horizon | Default CIF | Prepayment CIF |
|---:|---:|---:|
| 12 months | 0.6864% | 8.8930% |
| 24 months | 1.7715% | 22.3853% |
| 36 months | 2.7882% | 33.4158% |
| 60 months | 4.0692% | 49.8580% |

These estimates represent competing-risk cumulative incidence probabilities rather than crude Default counts divided by the original cohort.

The crude lifetime Default proportion in the constructed cohort is approximately **3.23%**, but this quantity is not interpreted as a fixed-horizon probability of Default.

## Kaplan-Meier Benchmark

Kaplan–Meier is retained as a descriptive benchmark.

In this benchmark, Prepayment is treated as censoring. This allows the project to demonstrate why the naive transformation `1 - KM` should not be interpreted as the cumulative probability of Default when a competing event is present.

| Horizon | Naive 1-KM | Aalen–Johansen Default CIF | Gap |
|---:|---:|---:|---:|
| 12 months | 0.7515% | 0.6864% | 0.0651 pp |
| 24 months | 2.0518% | 1.7715% | 0.2803 pp |
| 36 months | 3.4501% | 2.7882% | 0.6619 pp |
| 60 months | 5.5786% | 4.0692% | 1.5094 pp |

The difference becomes larger as the horizon increases and more loans experience Prepayment.

Therefore, in the presence of competing Prepayment:

**1 - KM(Default) is not equal to the Default CIF.**

Aalen–Johansen is consequently used as the project's primary estimator of cumulative Default probability.

## Why 36 Months Is the Primary Horizon

The project reports results at 12, 24, 36, and 60 months, but uses **36 months as the primary comparative horizon**.

Using the March 2026 data cutoff:

- approximately **69.62%** of loans are administratively eligible for a complete 36-month observation window;
- approximately **50.29%** are administratively eligible for a complete 60-month observation window.

The 36-month horizon therefore provides a balance between observing meaningful Default/Prepayment dynamics and retaining more recent origination cohorts.

For 36-month vintage comparison:

- vintages 2016–2021 have complete administrative eligibility;
- vintage 2022 is approximately 99.95% eligible;
- vintage 2023 is only approximately 13.69% eligible;
- vintages 2024–2026 do not yet have a complete 36-month observation window.

Accordingly, **2016–2022 are used as the primary mature-vintage comparison set**, while 2023–2026 are explicitly identified as maturity-limited.

The project uses a **95% administrative-eligibility threshold** as a reporting convention for vintage comparability. This threshold is a project decision, not an official Freddie Mac or Aalen–Johansen standard.

The 60-month horizon is retained as a **sensitivity horizon** rather than the main vintage-comparison horizon.

## Vintage Analysis

Default cumulative incidence is estimated separately by origination vintage using the competing-risks framework.

| Vintage | Default CIF at 36 Months | Comparison Status |
|---:|---:|---|
| 2016 | 0.9950% | Mature |
| 2017 | 3.2268% | Mature |
| 2018 | 4.6462% | Mature |
| 2019 | 4.9184% | Mature |
| 2020 | 1.8222% | Mature |
| 2021 | 1.3481% | Mature |
| 2022 | 2.9851% | Mature |
| 2023–2026 | Maturity-limited | Not directly comparable |

Later-vintage estimates remain visible in the dashboard for transparency, but they should not be interpreted as directly comparable full 36-month outcomes.
## Credit Risk Drivers

The project evaluates five origination characteristics:

- Credit Score
- Original Loan-to-Value Ratio (LTV)
- Original Debt-to-Income Ratio (DTI)
- Original Interest Rate
- Original Loan Term

Continuous variables are rescaled to make the estimated hazard ratios easier to interpret:

| Variable | HR Interpretation |
|---|---|
| Credit Score | +10 points |
| Original LTV | +10 percentage points |
| Original DTI | +10 percentage points |
| Interest Rate | +1 percentage point |
| Loan Term | +12 months |

## Cause-Specific Cox Model

The baseline Default model uses a cause-specific Cox proportional-hazards specification:

**h_D(t | X) = h_D0(t) × exp(beta'X)**

For this model, loans experiencing Prepayment leave the Default risk set at their Prepayment time.

The estimated baseline cause-specific hazard ratios are:

| Variable | Hazard Ratio |
|---|---:|
| Credit Score (+10 points) | 0.8721 |
| Original LTV (+10 pp) | 1.0785 |
| Original DTI (+10 pp) | 1.0003 |
| Interest Rate (+1 pp) | 1.1339 |
| Loan Term (+12 months) | 1.0327 |

The model has a concordance index of approximately **0.7224**.

These hazard ratios describe associations with the instantaneous Default hazard among loans that remain event-free. They should not be interpreted as causal effects.

For example, the Credit Score hazard ratio below 1 indicates that higher Credit Score is associated with a lower cause-specific Default hazard, holding the included covariates constant.

The DTI estimate is close to 1 in the baseline specification and should not be interpreted as evidence of a strong association based on this model alone.

## Proportional-Hazards Diagnostics

Proportional-hazards diagnostics flag time dependence for the modeled covariates.

Because the dataset contains more than 500,000 loans, statistical tests can detect relatively small departures from proportionality.

The project therefore does not automatically discard the Cox model. Instead, the baseline Cox specification is retained as a summary model and supplemented with a time-varying analysis.

## Start-Stop Time-Varying Cox Model

A start-stop Cox specification is used to examine whether covariate associations change with loan age.

Each loan's observed duration is divided into time intervals. The model then interacts centered covariates with log time to obtain horizon-specific hazard-ratio profiles.

Selected results are:

| Variable | 12m | 24m | 36m | 60m |
|---|---:|---:|---:|---:|
| Credit Score (+10) | 0.8922 | 0.8860 | 0.8824 | 0.8778 |
| Original LTV (+10 pp) | 1.0773 | 1.0782 | 1.0787 | 1.0793 |
| Original DTI (+10 pp) | 1.0010 | 1.0009 | 1.0009 | 1.0008 |
| Interest Rate (+1 pp) | 1.0621 | 1.1814 | 1.2573 | 1.3599 |
| Loan Term (+12m) | 1.0319 | 1.0287 | 1.0269 | 1.0246 |

This implementation is interpreted as a **discretized start-stop time-varying coefficient specification**. It is used to examine changes in associations over loan age rather than to claim an exact continuous functional form.

An earlier exploratory static covariate-by-log-duration specification is retained in the analytical history as `scripts/14_time_varying_cox.py`, but it is superseded by the start-stop implementation in `scripts/17_valid_time_varying_cox.py` and is not used as the final dashboard result.

## Fine-Gray Competing-Risk Regression

The project also estimates a Fine-Gray model for Default.

The model targets the **subdistribution hazard** associated with the Default cumulative incidence function, allowing Prepayment to be incorporated as a competing event.

The estimated full-cohort subdistribution hazard ratios are:

| Variable | Subdistribution HR |
|---|---:|
| Credit Score (+10 points) | 0.8686 |
| Original LTV (+10 pp) | 1.0733 |
| Original DTI (+10 pp) | 1.0022 |
| Interest Rate (+1 pp) | 1.0294 |
| Loan Term (+12 months) | 1.0331 |

The Fine-Gray model is estimated on the full survival cohort of **512,490 loans**.

For computational feasibility, variance estimation is disabled in the current full-cohort implementation. Fine-Gray results are therefore reported as **point estimates only**.

No Fine-Gray standard errors, confidence intervals, or p-values are reported, and no statistical-significance claims are made from these estimates.

## Cox vs Fine-Gray Interpretation

The two regression frameworks answer different questions.

**Cause-Specific Cox**

Examines the instantaneous Default hazard among loans that are still event-free.

**Fine-Gray**

Examines covariate associations with the subdistribution hazard linked to cumulative Default incidence while accounting for competing Prepayment.

Therefore, a cause-specific hazard ratio and a Fine-Gray subdistribution hazard ratio should not be interpreted as interchangeable quantities.

Both models are retained because together they provide complementary views of mortgage Default risk.
## Macroeconomic Period Analysis

The project supplements the loan-level survival analysis with a descriptive examination of mortgage performance around two selected macroeconomic windows.

### COVID Recession Window

The COVID recession window is defined as:

**March 2020 – April 2020**

NBER identifies February 2020 as the monthly business-cycle peak and April 2020 as the trough. Under NBER dating conventions, the corresponding recession months are March and April 2020.

### Interest-Rate Shock Window

The rate-shock window is defined as:

**January 2022 – October 2023**

This is a **project-defined analytical window**, not an official recession or business-cycle classification.

The window is designed to capture the sharp transition from the low-rate environment at the beginning of 2022 through the period in which U.S. mortgage rates rose substantially, reaching approximately 7.8% by late October 2023.

## First 36-Month Shock Exposure

For each loan, the project examines whether the selected macroeconomic windows overlap its first 36 observed performance months.

Loans are assigned to four descriptive exposure groups:

- COVID only
- Rate shock only
- Both COVID and rate shock
- No selected shock

The resulting exposure counts are:

| Exposure Group | Loans |
|---|---:|
| COVID and Rate Shock | 28,580 |
| COVID Only | 103,052 |
| Rate Shock Only | 168,011 |
| No Selected Shock | 212,847 |

Selected first-36-month outcomes are:

| Exposure Group | Default <=36m | Prepayment <=36m |
|---|---:|---:|
| COVID and Rate Shock | 7.4388% | 9.7446% |
| COVID Only | 4.3211% | 53.4119% |
| Rate Shock Only | 2.0528% | 14.3169% |
| No Selected Shock | 0.7794% | 27.8806% |

These percentages are **descriptive cohort outcomes**, not Aalen–Johansen cumulative incidence estimates. They should therefore not be labeled or interpreted as competing-risk PD estimates.

## Before / During / After Analysis

The project also evaluates monthly loan-performance behavior before, during, and after each selected macroeconomic window.

Metrics include:

- 30+ day delinquency rate
- 60+ day delinquency rate
- 90+ day delinquency rate
- Default event rate
- Prepayment event rate

These rates use **observed loan-months as the denominator**.

They are monthly performance rates rather than cumulative Default probabilities.

For the COVID window, the comparison periods are:

- Before: January 2020 – February 2020
- During: March 2020 – April 2020
- After: May 2020 – June 2020

For the rate-shock analysis:

- Before: March 2020 – December 2021
- During: January 2022 – October 2023
- After: November 2023 – August 2025

The rate-shock before-period overlaps the beginning of the COVID period. This overlap is retained explicitly and should not be interpreted as an isolated counterfactual baseline.

## Interpretation of Macro Results

The macroeconomic analysis is **descriptive, not causal**.

Loans are not randomly assigned to macroeconomic exposure groups. Origination vintage, borrower characteristics, loan age, refinancing incentives, selection effects, and other economic conditions can differ substantially across groups and periods.

Consequently, differences in delinquency, Default, or Prepayment behavior should be interpreted as observed associations rather than evidence that a selected macroeconomic window caused the outcomes.

## External Reference Basis

The project uses the following external sources to define and document the analytical context:

- **Freddie Mac — Single-Family Loan-Level Dataset General User Guide**  
  https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf

- **National Bureau of Economic Research — Business Cycle Dating Procedure: Frequently Asked Questions**  
  https://www.nber.org/research/business-cycle-dating/business-cycle-dating-procedure-frequently-asked-questions

- **National Bureau of Economic Research — Determination of the April 2020 Trough in U.S. Economic Activity**  
  https://www.nber.org/news/business-cycle-dating-committee-announcement-july-19-2021

- **Freddie Mac Research — More Borrowers Pay Discount Points, But It May Not Be Worth It**  
  https://www.freddiemac.com/research/insight/discount-points

The COVID recession dates follow NBER chronology. The January 2022–October 2023 rate-shock interval remains a project-defined research window informed by the documented rise in mortgage rates rather than an official economic-cycle classification.
## Analytical Pipeline

The project is organized as a sequential analytical pipeline:

```text
01   Data ingestion and schema audit
02   Initial survival construction checks
03   Delinquency audit
04   Default / Prepayment chronology audit
05   Survival eligibility audit
06   Final loan-level survival dataset
07   Follow-up horizon audit
08   Vintage horizon eligibility
08a  Raw performance-period audit
08b  Survival time-origin audit
08c  Loan Age sequence audit
09   Aalen–Johansen competing-risk CIF
10   Vintage competing-risk analysis
11   First-36-month macro shock exposure
12   Before / during / after shock analysis
13   Cause-specific Cox model
14   Exploratory time interaction model
15   Fine-Gray competing-risk regression
16   Dashboard summary datasets
17   Start-stop time-varying Cox model
18   Kaplan-Meier descriptive benchmark