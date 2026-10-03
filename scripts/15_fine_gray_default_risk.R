# ============================================================
# MORTGAGE CREDIT RISK PORTFOLIO
# STEP 15 - FINE-GRAY DEFAULT SUBDISTRIBUTION HAZARD MODEL
# FAST FULL-COHORT ESTIMATION
# ============================================================


# ------------------------------------------------------------
# 1. PACKAGE CHECK
# ------------------------------------------------------------

if (!requireNamespace("cmprsk", quietly = TRUE)) {
  stop("Package 'cmprsk' is not installed.")
}


# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

input_file <- file.path(
  "data",
  "processed",
  "survival_loan_level.csv"
)

output_model <- file.path(
  "outputs",
  "fine_gray_default_risk_drivers.csv"
)

output_audit <- file.path(
  "outputs",
  "fine_gray_model_sample_audit.csv"
)


# ------------------------------------------------------------
# 3. CONFIGURATION
# ------------------------------------------------------------

DEFAULT_EVENT <- 1
PREPAY_EVENT <- 2
CENSOR_EVENT <- 0


# ------------------------------------------------------------
# 4. LOAD DATA
# ------------------------------------------------------------

cat("\nMORTGAGE CREDIT RISK\n")
cat("STEP 15 - FINE-GRAY DEFAULT RISK MODEL\n")
cat("FAST FULL-COHORT ESTIMATION\n\n")

if (!file.exists(input_file)) {
  stop(
    paste(
      "Input file not found:",
      input_file
    )
  )
}

df <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

cat(
  "Survival cohort loaded:",
  format(
    nrow(df),
    big.mark = ","
  ),
  "\n"
)


# ------------------------------------------------------------
# 5. REQUIRED VARIABLES
# ------------------------------------------------------------

required_columns <- c(
  "duration_months",
  "event_code",
  "fico",
  "original_ltv",
  "original_dti",
  "original_interest_rate",
  "original_loan_term"
)

missing_columns <- setdiff(
  required_columns,
  names(df)
)

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}


# ------------------------------------------------------------
# 6. NUMERIC CONVERSION
# ------------------------------------------------------------

for (column in required_columns) {
  df[[column]] <- suppressWarnings(
    as.numeric(
      df[[column]]
    )
  )
}


# ------------------------------------------------------------
# 7. PRE-MODEL AUDIT
# ------------------------------------------------------------

total_cohort <- nrow(df)

missing_duration <- sum(
  is.na(df$duration_months)
)

missing_event <- sum(
  is.na(df$event_code)
)

missing_fico <- sum(
  is.na(df$fico)
)

missing_ltv <- sum(
  is.na(df$original_ltv)
)

missing_dti <- sum(
  is.na(df$original_dti)
)

missing_rate <- sum(
  is.na(df$original_interest_rate)
)

missing_term <- sum(
  is.na(df$original_loan_term)
)


# ------------------------------------------------------------
# 8. COMPLETE-CASE SAMPLE
# ------------------------------------------------------------

model_df <- df[
  complete.cases(
    df[
      required_columns
    ]
  ),
]

model_df <- model_df[
  model_df$duration_months >= 0,
]

valid_event <- (
  model_df$event_code
  %in%
  c(
    CENSOR_EVENT,
    DEFAULT_EVENT,
    PREPAY_EVENT
  )
)

if (!all(valid_event)) {
  stop(
    "Invalid event code detected."
  )
}


# ------------------------------------------------------------
# 9. FINE-GRAY TIME
# ------------------------------------------------------------
#
# duration = 0 is a valid first-observed-month outcome.
# For model estimation only, these observations are placed
# at 0.5 month so that all Fine-Gray times are positive.
# ------------------------------------------------------------

model_df$fg_duration <- (
  model_df$duration_months
)

zero_duration_count <- sum(
  model_df$fg_duration == 0
)

model_df$fg_duration[
  model_df$fg_duration == 0
] <- 0.5


# ------------------------------------------------------------
# 10. SCALE COVARIATES
# ------------------------------------------------------------

model_df$fico_10 <- (
  model_df$fico / 10
)

model_df$ltv_10 <- (
  model_df$original_ltv / 10
)

model_df$dti_10 <- (
  model_df$original_dti / 10
)

model_df$interest_rate_1pct <- (
  model_df$original_interest_rate
)

model_df$loan_term_12m <- (
  model_df$original_loan_term / 12
)


# ------------------------------------------------------------
# 11. PLAUSIBILITY AUDIT
# ------------------------------------------------------------

fico_outside <- sum(
  model_df$fico < 300 |
  model_df$fico > 850
)

ltv_outside <- sum(
  model_df$original_ltv < 0 |
  model_df$original_ltv > 200
)

dti_outside <- sum(
  model_df$original_dti < 0 |
  model_df$original_dti > 100
)

rate_nonpositive <- sum(
  model_df$original_interest_rate <= 0
)

term_nonpositive <- sum(
  model_df$original_loan_term <= 0
)


# ------------------------------------------------------------
# 12. EVENT ACCOUNTING
# ------------------------------------------------------------

default_events <- sum(
  model_df$event_code
  == DEFAULT_EVENT
)

prepay_events <- sum(
  model_df$event_code
  == PREPAY_EVENT
)

censored_observations <- sum(
  model_df$event_code
  == CENSOR_EVENT
)

if (
  default_events
  + prepay_events
  + censored_observations
  != nrow(model_df)
) {
  stop(
    "Event accounting failed."
  )
}


# ------------------------------------------------------------
# 13. MODEL MATRIX
# ------------------------------------------------------------

covariate_names <- c(
  "fico_10",
  "ltv_10",
  "dti_10",
  "interest_rate_1pct",
  "loan_term_12m"
)

covariate_labels <- c(
  "FICO (+10 points)",
  "Original LTV (+10 percentage points)",
  "Original DTI (+10 percentage points)",
  "Original interest rate (+1 percentage point)",
  "Original loan term (+12 months)"
)

covariate_matrix <- as.matrix(
  model_df[
    covariate_names
  ]
)

storage.mode(
  covariate_matrix
) <- "double"


# ------------------------------------------------------------
# 14. FIT FINE-GRAY MODEL
# ------------------------------------------------------------
#
# IMPORTANT PERFORMANCE DECISION
#
# cmprsk::crr() with variance=TRUE was prohibitively slow
# on the full 512k-loan cohort.
#
# This step therefore estimates the Fine-Gray coefficients
# on the FULL cohort with variance=FALSE.
#
# No sampling is performed.
#
# Statistical inference (SE / CI / p-value) is intentionally
# not reported by this fast estimation step.
# ------------------------------------------------------------

cat("\nFine-Gray specification:\n")
cat("  Failure of interest : Default (1)\n")
cat("  Competing event     : Prepayment (2)\n")
cat("  Censor              : 0\n")
cat("  Cohort strategy     : FULL COHORT - NO SAMPLING\n")
cat("  Variance estimation : FALSE\n\n")

cat(
  "Model observations:",
  format(
    nrow(model_df),
    big.mark = ","
  ),
  "\n"
)

cat(
  "Default events:",
  format(
    default_events,
    big.mark = ","
  ),
  "\n"
)

cat(
  "Prepayment events:",
  format(
    prepay_events,
    big.mark = ","
  ),
  "\n\n"
)

cat(
  "Fitting Fine-Gray coefficients...\n"
)

start_time <- proc.time()

fg_model <- cmprsk::crr(
  ftime = model_df$fg_duration,
  fstatus = model_df$event_code,
  cov1 = covariate_matrix,
  failcode = DEFAULT_EVENT,
  cencode = CENSOR_EVENT,
  variance = FALSE
)

elapsed <- (
  proc.time()
  - start_time
)[["elapsed"]]


# ------------------------------------------------------------
# 15. EXTRACT COEFFICIENTS
# ------------------------------------------------------------

beta <- as.numeric(
  fg_model$coef
)

shr <- exp(
  beta
)

model_output <- data.frame(
  variable = covariate_names,
  label = covariate_labels,
  coefficient = beta,
  subdistribution_hazard_ratio = shr,
  inference_available = FALSE,
  stringsAsFactors = FALSE
)


# ------------------------------------------------------------
# 16. SAMPLE AUDIT
# ------------------------------------------------------------

audit_output <- data.frame(
  metric = c(
    "Total survival cohort",
    "Complete-case Fine-Gray sample",
    "Excluded from complete-case model",
    "Default events",
    "Prepayment events",
    "Censored observations",
    "Zero-duration observations adjusted to 0.5",
    "Missing duration",
    "Missing event code",
    "Missing FICO",
    "Missing original LTV",
    "Missing original DTI",
    "Missing original interest rate",
    "Missing original loan term",
    "FICO outside 300-850",
    "LTV outside 0-200",
    "DTI outside 0-100",
    "Interest rate nonpositive",
    "Loan term nonpositive",
    "Fine-Gray variance estimation",
    "Fine-Gray runtime seconds"
  ),
  value = c(
    total_cohort,
    nrow(model_df),
    total_cohort - nrow(model_df),
    default_events,
    prepay_events,
    censored_observations,
    zero_duration_count,
    missing_duration,
    missing_event,
    missing_fico,
    missing_ltv,
    missing_dti,
    missing_rate,
    missing_term,
    fico_outside,
    ltv_outside,
    dti_outside,
    rate_nonpositive,
    term_nonpositive,
    0,
    elapsed
  ),
  stringsAsFactors = FALSE
)


# ------------------------------------------------------------
# 17. VALIDATION
# ------------------------------------------------------------

validation_errors <- character()

if (
  length(beta)
  != length(covariate_names)
) {
  validation_errors <- c(
    validation_errors,
    "Unexpected coefficient count"
  )
}

if (
  any(
    !is.finite(beta)
  )
) {
  validation_errors <- c(
    validation_errors,
    "Non-finite coefficient detected"
  )
}

if (
  any(
    !is.finite(shr)
  )
) {
  validation_errors <- c(
    validation_errors,
    "Non-finite sHR detected"
  )
}

if (
  any(
    shr <= 0
  )
) {
  validation_errors <- c(
    validation_errors,
    "Non-positive sHR detected"
  )
}

if (
  nrow(model_df)
  != total_cohort
) {
  cat(
    "\nNOTE:",
    total_cohort - nrow(model_df),
    "observations excluded by complete-case requirements.\n"
  )
}


# ------------------------------------------------------------
# 18. PRINT RESULTS
# ------------------------------------------------------------

cat(
  "\n",
  paste(
    rep(
      "=",
      100
    ),
    collapse = ""
  ),
  "\n",
  sep = ""
)

cat(
  "FINE-GRAY DEFAULT SUBDISTRIBUTION HAZARD MODEL\n"
)

cat(
  paste(
    rep(
      "=",
      100
    ),
    collapse = ""
  ),
  "\n",
  sep = ""
)

cat(
  sprintf(
    "%-52s %14s %18s\n",
    "Variable",
    "Coefficient",
    "sHR"
  )
)

cat(
  paste(
    rep(
      "-",
      100
    ),
    collapse = ""
  ),
  "\n",
  sep = ""
)

for (
  i in seq_len(
    nrow(model_output)
  )
) {

  cat(
    sprintf(
      "%-52s %14.6f %18.4f\n",
      model_output$label[i],
      model_output$coefficient[i],
      model_output$subdistribution_hazard_ratio[i]
    )
  )
}


# ------------------------------------------------------------
# 19. DIAGNOSTICS
# ------------------------------------------------------------

cat(
  "\nModel diagnostics:\n"
)

cat(
  sprintf(
    "  Observations       : %s\n",
    format(
      nrow(model_df),
      big.mark = ","
    )
  )
)

cat(
  sprintf(
    "  Default events     : %s\n",
    format(
      default_events,
      big.mark = ","
    )
  )
)

cat(
  sprintf(
    "  Prepayment events  : %s\n",
    format(
      prepay_events,
      big.mark = ","
    )
  )
)

cat(
  sprintf(
    "  Converged          : %s\n",
    as.character(
      fg_model$converged
    )
  )
)

cat(
  sprintf(
    "  Runtime            : %.2f seconds\n",
    elapsed
  )
)

cat(
  "  Variance / CI / p  : NOT ESTIMATED IN FAST FULL-COHORT RUN\n"
)


# ------------------------------------------------------------
# 20. INTERPRETATION
# ------------------------------------------------------------

cat(
  "\nInterpretation:\n"
)

cat(
  paste0(
    "  sHR > 1 indicates a higher Default ",
    "subdistribution hazard after adjustment ",
    "for the other included covariates.\n"
  )
)

cat(
  paste0(
    "  sHR < 1 indicates a lower Default ",
    "subdistribution hazard.\n"
  )
)

cat(
  paste0(
    "  Prepayment is explicitly retained as a ",
    "competing event.\n"
  )
)

cat(
  paste0(
    "  sHR is not the same estimand as the ",
    "cause-specific Cox hazard ratio.\n"
  )
)

cat(
  paste0(
    "  This fast full-cohort run reports point ",
    "estimates only; it does not report ",
    "standard errors, confidence intervals, ",
    "or p-values.\n"
  )
)

cat(
  "  Results represent associations, not causal effects.\n"
)


# ------------------------------------------------------------
# 21. VALIDATION REPORT
# ------------------------------------------------------------

cat(
  "\n",
  paste(
    rep(
      "=",
      72
    ),
    collapse = ""
  ),
  "\n",
  sep = ""
)

cat("VALIDATION\n")

cat(
  paste(
    rep(
      "=",
      72
    ),
    collapse = ""
  ),
  "\n",
  sep = ""
)

cat(
  "Validation errors:",
  length(
    validation_errors
  ),
  "\n"
)

if (
  length(
    validation_errors
  ) > 0
) {

  for (
    error in validation_errors
  ) {

    cat(
      " ",
      error,
      "\n"
    )
  }

  stop(
    "Step 15 validation failed."
  )
}


# ------------------------------------------------------------
# 22. SAVE OUTPUTS
# ------------------------------------------------------------

dir.create(
  "outputs",
  showWarnings = FALSE,
  recursive = TRUE
)

write.csv(
  model_output,
  output_model,
  row.names = FALSE
)

write.csv(
  audit_output,
  output_audit,
  row.names = FALSE
)

cat(
  "\nModel output saved to:\n",
  output_model,
  "\n",
  sep = ""
)

cat(
  "\nSample audit saved to:\n",
  output_audit,
  "\n",
  sep = ""
)

cat(
  "\nStep 15 complete.\n"
)