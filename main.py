import math
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EHA_PANEL_PATH = os.path.join(DATA_DIR, "refinement_5_eha_panel.csv")
MODEL_PANEL_PATH = os.path.join(DATA_DIR, "refinement_6_eha_model_panel.csv")
COEFFICIENTS_PATH = os.path.join(DATA_DIR, "refinement_6_eha_logit_coefficients.csv")

TARGET = "adopted_this_year"
LEARNING_MECHANISM = "proportion_state_population_local_restriction"
COVARIATES = [
    "ordinance_profile_population",
    "municipal_population",
    "percent_25_plus_high_school_or_higher",
    "non_hispanic_white_share",
    "per_capita_income",
    "per_capita_spending",
    "texas_smoker_percentage",
    LEARNING_MECHANISM,
]


def main():
    panel = load_model_panel()
    fit = fit_logit_mle(panel, TARGET, COVARIATES)
    coefficients = build_coefficient_table(fit)

    panel.to_csv(MODEL_PANEL_PATH, index=False)
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)

    print_model_summary(panel, fit, coefficients)


def load_model_panel():
    panel = pd.read_csv(EHA_PANEL_PATH)
    model_columns = [
        "Municipality",
        "County",
        "Year",
        "Adoption Year",
        TARGET,
        *COVARIATES,
    ]
    panel = panel[model_columns].copy()

    for column in [TARGET, *COVARIATES]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    before_rows = len(panel)
    panel = panel.dropna(subset=[TARGET, *COVARIATES]).copy()
    dropped_rows = before_rows - len(panel)
    dropped_municipalities = sorted(
        set(pd.read_csv(EHA_PANEL_PATH).loc[
            lambda df: df[COVARIATES].isna().any(axis=1),
            "Municipality",
        ])
    )

    panel.attrs["dropped_rows"] = dropped_rows
    panel.attrs["dropped_municipalities"] = dropped_municipalities
    return panel


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
    standard_errors = np.sqrt(np.diag(covariance))
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


def build_coefficient_table(fit):
    return pd.DataFrame(
        {
            "term": fit["names"],
            "coefficient_standardized": fit["beta_standardized"],
            "standard_error_standardized": fit["standard_errors_standardized"],
            "z": fit["z_scores"],
            "p_value": fit["p_values"],
            "coefficient_original_scale": fit["beta_original_scale"],
            "is_learning_mechanism": [
                term == LEARNING_MECHANISM for term in fit["names"]
            ],
        }
    )


def print_model_summary(panel, fit, coefficients):
    learning_row = coefficients[
        coefficients["term"].eq(LEARNING_MECHANISM)
    ].iloc[0]

    print("Discrete-time logit hazard model")
    print(f"Rows used: {fit['n']}")
    print(f"Municipalities used: {panel['Municipality'].nunique()}")
    print(f"Adoption events: {fit['events']}")
    print(f"Log likelihood: {fit['log_likelihood']:.4f}")
    print(f"Converged: {fit['converged']} in {fit['iterations']} iterations")
    print(f"Clean model panel: {MODEL_PANEL_PATH}")
    print(f"Coefficient output: {COEFFICIENTS_PATH}")
    print()
    print("Dropped incomplete rows:", panel.attrs["dropped_rows"])
    print("Dropped municipalities:", ", ".join(panel.attrs["dropped_municipalities"]))
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
