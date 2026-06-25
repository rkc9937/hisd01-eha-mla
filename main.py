import math
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EHA_PANEL_PATH = os.path.join(DATA_DIR, "refinement_5_eha_panel.csv")
MODEL_PANEL_PATH = os.path.join(DATA_DIR, "refinement_6_eha_model_panel.csv")
COEFFICIENTS_PATH = os.path.join(DATA_DIR, "refinement_6_eha_logit_coefficients.csv")
KENTUCKY_EHA_PANEL_PATH = os.path.join(DATA_DIR, "kentucky_refinement_5_eha_panel.csv")
KENTUCKY_MODEL_PANEL_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_6_eha_model_panel.csv",
)
KENTUCKY_COEFFICIENTS_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_6_eha_logit_coefficients.csv",
)
KENTUCKY_DIAGNOSTICS_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_6_eha_model_diagnostics.csv",
)
KENTUCKY_SENSITIVITY_MODEL_PANEL_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_6_eha_model_panel_sensitivity.csv",
)
KENTUCKY_COMPLETE_MODEL_PANEL_PATH = os.path.join(
    DATA_DIR,
    "kentucky_refinement_6_eha_model_panel_complete.csv",
)
KENTUCKY_SMOKER_PATH = os.path.join(DATA_DIR, "kentucky_smoker_percentage_year.csv")
KENTUCKY_IDEOLOGY_PATH = os.path.join(DATA_DIR, "kentucky_ideology_per_year.csv")
KENTUCKY_TOBACCO_LOBBYING_PATH = os.path.join(
    DATA_DIR,
    "kentucky_tobacco_lobbying_2009_2023.csv",
)

TARGET = "adopted_this_year"
LEARNING_MECHANISM = "proportion_state_population_local_restriction"
TEXAS_COVARIATES = [
    "ordinance_profile_population",
    "municipal_population",
    "percent_25_plus_high_school_or_higher",
    "non_hispanic_white_share",
    "per_capita_income",
    "per_capita_spending",
    "texas_smoker_percentage",
    LEARNING_MECHANISM,
]
KENTUCKY_COVARIATES = [
    "municipal_population",
    "percent_25_plus_high_school_or_higher",
    "non_hispanic_white_share",
    "per_capita_income",
    "per_capita_spending",
    "kentucky_smoker_percentage",
    "nominate_state_government_ideology",
    "tobacco_lobbyist_ratio",
    LEARNING_MECHANISM,
]
KENTUCKY_COMPLETE_COVARIATES = [
    "municipal_population",
    "percent_25_plus_high_school_or_higher",
    "non_hispanic_white_share",
    "per_capita_income",
    "per_capita_spending",
    "kentucky_smoker_percentage",
    LEARNING_MECHANISM,
]
KENTUCKY_SENSITIVITY_COVARIATES = [
    "municipal_population",
    "percent_25_plus_high_school_or_higher",
    "non_hispanic_white_share",
    "per_capita_income",
    "per_capita_spending",
    "kentucky_smoker_percentage",
    "nominate_state_government_ideology",
    "tobacco_lobbyist_ratio",
    "tobacco_lobbyist_ratio_missing",
    LEARNING_MECHANISM,
]


def main():
    if os.path.exists(COEFFICIENTS_PATH):
        run_kentucky_models()
        return

    run_model(
        "Texas",
        EHA_PANEL_PATH,
        MODEL_PANEL_PATH,
        COEFFICIENTS_PATH,
        TEXAS_COVARIATES,
    )


def run_model(state_name, panel_path, model_panel_path, coefficients_path, covariates):
    source_panel = pd.read_csv(panel_path)
    panel, fit, coefficients = fit_model_from_source_panel(source_panel, covariates)
    panel.to_csv(model_panel_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)

    print_model_summary(
        state_name,
        panel,
        fit,
        coefficients,
        model_panel_path,
        coefficients_path,
    )


def run_kentucky_models():
    augmented_panel = build_kentucky_augmented_panel()

    complete_panel, complete_fit, complete_coefficients = fit_model_from_source_panel(
        augmented_panel,
        KENTUCKY_COMPLETE_COVARIATES,
    )
    complete_coefficients.insert(0, "model", "complete")
    complete_coefficients["n"] = complete_fit["n"]
    complete_coefficients["events"] = complete_fit["events"]
    complete_coefficients["log_likelihood"] = complete_fit["log_likelihood"]
    complete_coefficients["converged"] = complete_fit["converged"]

    primary_source_panel = augmented_panel.copy()
    primary_source_panel["tobacco_lobbyist_ratio"] = primary_source_panel[
        "tobacco_lobbyist_ratio_primary"
    ]
    primary_panel, primary_fit, primary_coefficients = fit_model_from_source_panel(
        primary_source_panel,
        KENTUCKY_COVARIATES,
    )
    primary_coefficients.insert(0, "model", "primary")
    primary_coefficients["n"] = primary_fit["n"]
    primary_coefficients["events"] = primary_fit["events"]
    primary_coefficients["log_likelihood"] = primary_fit["log_likelihood"]
    primary_coefficients["converged"] = primary_fit["converged"]

    sensitivity_source_panel = augmented_panel.copy()
    sensitivity_source_panel["tobacco_lobbyist_ratio"] = sensitivity_source_panel[
        "tobacco_lobbyist_ratio_sensitivity"
    ]
    (
        sensitivity_panel,
        sensitivity_fit,
        sensitivity_coefficients,
    ) = fit_model_from_source_panel(
        sensitivity_source_panel,
        KENTUCKY_SENSITIVITY_COVARIATES,
    )
    sensitivity_coefficients.insert(0, "model", "sensitivity")
    sensitivity_coefficients["n"] = sensitivity_fit["n"]
    sensitivity_coefficients["events"] = sensitivity_fit["events"]
    sensitivity_coefficients["log_likelihood"] = sensitivity_fit["log_likelihood"]
    sensitivity_coefficients["converged"] = sensitivity_fit["converged"]

    combined_coefficients = pd.concat(
        [complete_coefficients, primary_coefficients, sensitivity_coefficients],
        ignore_index=True,
    )
    diagnostics = pd.DataFrame(
        [
            build_diagnostics_row("complete", complete_panel, complete_fit),
            build_diagnostics_row("primary", primary_panel, primary_fit),
            build_diagnostics_row("sensitivity", sensitivity_panel, sensitivity_fit),
        ]
    )

    complete_panel.to_csv(KENTUCKY_COMPLETE_MODEL_PANEL_PATH, index=False)
    primary_panel.to_csv(KENTUCKY_MODEL_PANEL_PATH, index=False)
    sensitivity_panel.to_csv(KENTUCKY_SENSITIVITY_MODEL_PANEL_PATH, index=False)
    combined_coefficients.to_csv(KENTUCKY_COEFFICIENTS_PATH, index=False)
    diagnostics.to_csv(KENTUCKY_DIAGNOSTICS_PATH, index=False)

    print_model_summary(
        "Kentucky complete",
        complete_panel,
        complete_fit,
        complete_coefficients,
        KENTUCKY_COMPLETE_MODEL_PANEL_PATH,
        KENTUCKY_COEFFICIENTS_PATH,
    )
    print()
    print_model_summary(
        "Kentucky primary",
        primary_panel,
        primary_fit,
        primary_coefficients,
        KENTUCKY_MODEL_PANEL_PATH,
        KENTUCKY_COEFFICIENTS_PATH,
    )
    print()
    print_model_summary(
        "Kentucky sensitivity",
        sensitivity_panel,
        sensitivity_fit,
        sensitivity_coefficients,
        KENTUCKY_SENSITIVITY_MODEL_PANEL_PATH,
        KENTUCKY_COEFFICIENTS_PATH,
    )


def fit_model_from_source_panel(source_panel, covariates):
    panel, model_covariates = load_model_panel(source_panel, covariates)
    fit = fit_logit_mle(panel, TARGET, model_covariates)
    coefficients = build_coefficient_table(fit, LEARNING_MECHANISM)
    return panel, fit, coefficients


def build_diagnostics_row(model_name, panel, fit, threshold=0.5):
    y = panel[TARGET].to_numpy(dtype=int)
    probabilities = predicted_probabilities(panel, fit)
    predictions = (probabilities >= threshold).astype(int)

    true_positives = int(((predictions == 1) & (y == 1)).sum())
    false_positives = int(((predictions == 1) & (y == 0)).sum())
    true_negatives = int(((predictions == 0) & (y == 0)).sum())
    false_negatives = int(((predictions == 0) & (y == 1)).sum())

    return {
        "model": model_name,
        "rows": len(panel),
        "municipalities": panel["Municipality"].nunique(),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "threshold": threshold,
        "accuracy_at_0_5": float((predictions == y).mean()),
        "auc": auc_score(y, probabilities),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "mean_predicted_hazard": float(probabilities.mean()),
        "mean_predicted_event_hazard": float(probabilities[y == 1].mean()),
        "mean_predicted_non_event_hazard": float(probabilities[y == 0].mean()),
        "log_likelihood": fit["log_likelihood"],
        "converged": fit["converged"],
    }


def predicted_probabilities(panel, fit):
    covariates = fit["names"][1:]
    raw_x = panel[covariates].to_numpy(dtype=float)
    standardized_x = (raw_x - fit["means"]) / fit["stds"]
    x = np.column_stack([np.ones(len(panel)), standardized_x])
    return sigmoid(x @ fit["beta_standardized"])


def auc_score(y, probabilities):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return np.nan

    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positive_rank_sum = ranks[y == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def build_kentucky_augmented_panel():
    panel = pd.read_csv(KENTUCKY_EHA_PANEL_PATH)

    panel = panel.drop(
        columns=[
            column
            for column in [
                "kentucky_smoker_percentage",
                "nominate_state_government_ideology",
                "tobacco_lobbyist_registrations",
                "total_lobbyist_registrations",
                "tobacco_lobbyist_ratio",
                "tobacco_lobbyist_ratio_primary",
                "tobacco_lobbyist_ratio_sensitivity",
                "tobacco_lobbyist_ratio_missing",
            ]
            if column in panel.columns
        ]
    )

    panel = panel.merge(kentucky_smoker_by_year(), on="Year", how="left")
    panel = panel.merge(kentucky_ideology_by_year(panel["Year"]), on="Year", how="left")
    panel = panel.merge(kentucky_tobacco_lobbying_by_year(), on="Year", how="left")
    return panel


def kentucky_smoker_by_year():
    smoker = pd.read_csv(KENTUCKY_SMOKER_PATH)
    return smoker.rename(
        columns={
            "Year": "Year",
            "Percentage": "kentucky_smoker_percentage",
        }
    )[["Year", "kentucky_smoker_percentage"]]


def kentucky_ideology_by_year(panel_years):
    year_frame = pd.DataFrame(
        {"Year": range(int(panel_years.min()), int(panel_years.max()) + 1)}
    )
    ideology = pd.read_csv(KENTUCKY_IDEOLOGY_PATH).rename(columns={"year": "Year"})
    ideology = year_frame.merge(ideology, on="Year", how="left").sort_values("Year")
    ideology["nominate_state_government_ideology"] = (
        pd.to_numeric(
            ideology["nominate_state_government_ideology"],
            errors="coerce",
        )
        .interpolate()
        .ffill()
        .bfill()
    )
    return ideology


def kentucky_tobacco_lobbying_by_year():
    lobbying = pd.read_csv(KENTUCKY_TOBACCO_LOBBYING_PATH)
    lobbying = lobbying.rename(columns={"year": "Year"}).sort_values("Year")
    for column in [
        "tobacco_lobbyist_registrations",
        "total_lobbyist_registrations",
        "tobacco_lobbyist_ratio",
    ]:
        lobbying[column] = pd.to_numeric(lobbying[column], errors="coerce")

    observed_ratio = lobbying["tobacco_lobbyist_ratio"]
    lobbying["tobacco_lobbyist_ratio_primary"] = observed_ratio.ffill()
    lobbying["tobacco_lobbyist_ratio_sensitivity"] = observed_ratio.ffill()
    lobbying["tobacco_lobbyist_ratio_sensitivity"] = lobbying[
        "tobacco_lobbyist_ratio_sensitivity"
    ].fillna(observed_ratio.mean())
    lobbying["tobacco_lobbyist_ratio_missing"] = observed_ratio.isna().astype(int)

    return lobbying[
        [
            "Year",
            "tobacco_lobbyist_registrations",
            "total_lobbyist_registrations",
            "tobacco_lobbyist_ratio_primary",
            "tobacco_lobbyist_ratio_sensitivity",
            "tobacco_lobbyist_ratio_missing",
        ]
    ]


def load_model_panel(panel_path, covariates):
    if isinstance(panel_path, pd.DataFrame):
        source_panel = panel_path.copy()
    else:
        source_panel = pd.read_csv(panel_path)

    requested_covariates = list(covariates)
    missing_columns = [
        covariate
        for covariate in requested_covariates
        if covariate not in source_panel.columns
    ]
    if missing_columns:
        raise RuntimeError(f"Missing model covariates: {missing_columns}")

    omitted_covariates = [
        covariate
        for covariate in requested_covariates
        if covariate != LEARNING_MECHANISM and source_panel[covariate].isna().all()
    ]
    covariates = [
        covariate
        for covariate in requested_covariates
        if covariate == LEARNING_MECHANISM
        or not source_panel[covariate].isna().all()
    ]
    model_columns = [
        "Municipality",
        "Year",
        "Adoption Year",
        TARGET,
        *covariates,
    ]
    if "County" in source_panel.columns:
        model_columns.insert(1, "County")
    if "geography_type" in source_panel.columns:
        model_columns.insert(1, "geography_type")
    model_columns = [
        column for column in model_columns if column in source_panel.columns
    ]
    panel = source_panel[model_columns].copy()

    for column in [TARGET, *covariates]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    before_rows = len(panel)
    panel = panel.dropna(subset=[TARGET, *covariates]).copy()
    dropped_rows = before_rows - len(panel)

    zero_variance_covariates = [
        covariate
        for covariate in covariates
        if panel[covariate].nunique(dropna=True) <= 1
    ]
    if zero_variance_covariates:
        covariates = [
            covariate
            for covariate in covariates
            if covariate not in zero_variance_covariates
        ]
        panel = panel.drop(columns=zero_variance_covariates)

    missing_covariates = [
        covariate
        for covariate in covariates
        if source_panel[covariate].isna().any()
    ]
    if missing_covariates:
        missing_mask = source_panel[missing_covariates].isna().any(axis=1)
        dropped_municipalities = sorted(
            set(source_panel.loc[missing_mask, "Municipality"])
        )
    else:
        dropped_municipalities = []

    panel.attrs["dropped_rows"] = dropped_rows
    panel.attrs["dropped_municipalities"] = dropped_municipalities
    panel.attrs["omitted_covariates"] = omitted_covariates
    panel.attrs["zero_variance_covariates"] = zero_variance_covariates
    return panel, covariates


def fit_logit_mle(panel, target, covariates, max_iter=100, tolerance=1e-8):
    y = panel[target].to_numpy(dtype=float)
    raw_x = panel[covariates].to_numpy(dtype=float)
    means = raw_x.mean(axis=0)
    stds = raw_x.std(axis=0, ddof=0)
    stds = np.where(stds == 0, 1.0, stds)
    standardized_x = (raw_x - means) / stds
    x = np.column_stack([np.ones(len(panel)), standardized_x])
    names = ["intercept", *covariates]

    beta = np.zeros(x.shape[1])
    current_nll = negative_log_likelihood(x, y, beta)
    converged = False

    for iteration in range(1, max_iter + 1):
        eta = x @ beta
        p = sigmoid(eta)
        gradient = x.T @ (p - y)
        weights = p * (1 - p)
        hessian = (x.T * weights) @ x

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient

        step_size = 1.0
        while step_size > 1e-8:
            candidate = beta - step_size * step
            candidate_nll = negative_log_likelihood(x, y, candidate)
            if candidate_nll <= current_nll:
                break
            step_size *= 0.5

        beta = candidate
        improvement = current_nll - candidate_nll
        current_nll = candidate_nll

        if np.linalg.norm(step_size * step, ord=2) < tolerance or improvement < tolerance:
            converged = True
            break

    covariance = covariance_matrix(x, beta)
    covariance_diagonal = np.diag(covariance)
    standard_errors = np.sqrt(np.where(covariance_diagonal < 0, np.nan, covariance_diagonal))
    z_scores = beta / standard_errors
    p_values = np.array([two_sided_normal_p_value(z) for z in z_scores])

    original_scale_beta = standardized_to_original_scale(beta, means, stds)

    return {
        "names": names,
        "beta_standardized": beta,
        "beta_original_scale": original_scale_beta,
        "standard_errors_standardized": standard_errors,
        "z_scores": z_scores,
        "p_values": p_values,
        "log_likelihood": -current_nll,
        "iterations": iteration,
        "converged": converged,
        "means": means,
        "stds": stds,
        "n": len(y),
        "events": int(y.sum()),
    }


def sigmoid(values):
    return np.where(
        values >= 0,
        1 / (1 + np.exp(-values)),
        np.exp(values) / (1 + np.exp(values)),
    )


def negative_log_likelihood(x, y, beta):
    eta = x @ beta
    return float(np.sum(np.logaddexp(0, eta) - y * eta))


def covariance_matrix(x, beta):
    p = sigmoid(x @ beta)
    weights = p * (1 - p)
    hessian = (x.T * weights) @ x
    return np.linalg.pinv(hessian)


def standardized_to_original_scale(beta, means, stds):
    original = beta.copy()
    original[1:] = beta[1:] / stds
    original[0] = beta[0] - np.sum(beta[1:] * means / stds)
    return original


def two_sided_normal_p_value(z_score):
    return math.erfc(abs(float(z_score)) / math.sqrt(2))


def build_coefficient_table(fit, learning_mechanism):
    return pd.DataFrame(
        {
            "term": fit["names"],
            "coefficient_standardized": fit["beta_standardized"],
            "standard_error_standardized": fit["standard_errors_standardized"],
            "z": fit["z_scores"],
            "p_value": fit["p_values"],
            "coefficient_original_scale": fit["beta_original_scale"],
            "is_learning_mechanism": [
                term == learning_mechanism for term in fit["names"]
            ],
        }
    )


def print_model_summary(
    state_name,
    panel,
    fit,
    coefficients,
    model_panel_path,
    coefficients_path,
):
    learning_row = coefficients[
        coefficients["term"].eq(LEARNING_MECHANISM)
    ].iloc[0]

    print(f"{state_name} discrete-time logit hazard model")
    print(f"Rows used: {fit['n']}")
    print(f"Municipalities used: {panel['Municipality'].nunique()}")
    print(f"Adoption events: {fit['events']}")
    print(f"Log likelihood: {fit['log_likelihood']:.4f}")
    print(f"Converged: {fit['converged']} in {fit['iterations']} iterations")
    print(f"Clean model panel: {model_panel_path}")
    print(f"Coefficient output: {coefficients_path}")
    print()
    print("Dropped incomplete rows:", panel.attrs["dropped_rows"])
    print("Dropped municipalities:", ", ".join(panel.attrs["dropped_municipalities"]))
    print("Omitted all-missing covariates:", ", ".join(panel.attrs["omitted_covariates"]))
    print(
        "Omitted zero-variance covariates:",
        ", ".join(panel.attrs["zero_variance_covariates"]),
    )
    print()
    print("Learning mechanism coefficient")
    print(
        learning_row[
            [
                "term",
                "coefficient_standardized",
                "standard_error_standardized",
                "z",
                "p_value",
                "coefficient_original_scale",
            ]
        ].to_string()
    )
    print()
    print(coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
