"""
Univariate and multivariate OLS regression of regret(T).

Univariate:  regret ~ a*f(T)   for f in {T, sqrt(T)}
Multivariate: regret ~ a*T + b*sqrt(T)  (both bases together)

Reports slope(s), intercept, R², adjusted R², p-values, and 95% CI on each
coefficient.  Traces (rc, ablation, UCRL2, UCRL3, KL) are fit separately.
"""

import pickle
import os
import numpy as np
from scipy import stats

# ── data loading ──────────────────────────────────────────────────────────────

def load_regret(folder, n_runs, baselines=False, normalize_regret=True, start_no=0):
    raw = {}
    for run_no in range(start_no, start_no + n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)

        def normalize(l):
            return np.array(l) / run_data["ideal_gain"] if normalize_regret else np.array(l)

        for key in ("rc", "ablation"):
            raw.setdefault(key, []).append(normalize(run_data[key]["regret"]))

        if baselines:
            baselines_file = f"{folder}/baselines_{run_no}"
            if os.path.isfile(baselines_file):
                with open(baselines_file, "rb") as f:
                    baseline_data = pickle.load(f)
            elif os.path.isfile(f"{folder}/baselines_run_{run_no}"):
                with open(f"{folder}/baselines_run_{run_no}", "rb") as f:
                    baseline_data = pickle.load(f)
            else:
                with open(f"{folder}/no_UCRL3_baselines_{run_no}", "rb") as f:
                    baseline_data = pickle.load(f)
                with open(f"{folder}/UCRL3_baselines_{run_no}", "rb") as f:
                    baseline_data |= pickle.load(f)
            for key in ("UCRL2", "UCRL3", "KL"):
                raw.setdefault(key, []).append(normalize(baseline_data[key]["regret"]))

    return {k: np.mean(v, axis=0) for k, v in raw.items()}


# ── OLS helpers ───────────────────────────────────────────────────────────────

def ols(X, y):
    """Return beta, R², adj-R², per-coefficient (stderr, t, p, ci_lo, ci_hi)."""
    n, p = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)

    s2 = ss_res / (n - p)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    t_stats = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n - p)
    t_crit = stats.t.ppf(0.975, df=n - p)
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se

    return beta, r2, adj_r2, se, t_stats, p_vals, ci_lo, ci_hi


def print_ols(coef_names, beta, r2, adj_r2, se, t_stats, p_vals, ci_lo, ci_hi):
    col = 16
    hdr = f"  {'Coef':<{col}}  {'value':>13}  {'SE':>10}  {'t':>8}  {'p-value':>10}  {'95% CI'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, name in enumerate(coef_names):
        ci_str = f"[{ci_lo[i]:+.4e}, {ci_hi[i]:+.4e}]"
        print(f"  {name:<{col}}  {beta[i]:>+13.4e}  {se[i]:>10.4e}  {t_stats[i]:>8.3f}  {p_vals[i]:>10.3e}  {ci_str}")
    print(f"  R² = {r2:.5f}   adj-R² = {adj_r2:.5f}")


# ── per-trace fit ─────────────────────────────────────────────────────────────

def fit_trace(T, regret):
    mask = T > 0
    T_m = T[mask]
    y = regret[mask]
    n = len(y)

    # ── univariate ────────────────────────────────────────────────────────────
    univariate = {
        "T":       T_m,
        "sqrt(T)": np.sqrt(T_m),
    }
    print("  [Univariate]")
    for basis_name, x in univariate.items():
        X = x.reshape(-1, 1)
        beta, r2, adj_r2, se, t_stats, p_vals, ci_lo, ci_hi = ols(X, y)
        coef_names = [f"a ({basis_name})"]
        print(f"    {basis_name}:")
        print_ols(coef_names, beta, r2, adj_r2, se, t_stats, p_vals, ci_lo, ci_hi)
        print()

    # ── multivariate ──────────────────────────────────────────────────────────
    print("  [Multivariate: a*T + b*sqrt(T)]")
    X_multi = np.column_stack([T_m, np.sqrt(T_m)])
    beta, r2, adj_r2, se, t_stats, p_vals, ci_lo, ci_hi = ols(X_multi, y)
    coef_names = ["a (T)", "b (sqrt(T))"]
    print_ols(coef_names, beta, r2, adj_r2, se, t_stats, p_vals, ci_lo, ci_hi)


# ── main ──────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("exp_out/11_states_2/", 50, True,  "11 states"),
    ("exp_out/21_states_2/", 50, True,  "21 states"),
    ("exp_out/51_states_2/", 50, True,  "51 states"),
]

TRACE_LABELS = {
    "rc":       "UCRL-TSAC",
    "ablation": "Ablation",
    "UCRL2":    "UCRL2",
    "UCRL3":    "UCRL3",
    "KL":       "KL-UCRL",
}

if __name__ == "__main__":
    T_all = np.array([x * 10000 for x in range(1000)], dtype=float)

    for folder, n_runs, baselines, exp_label in EXPERIMENTS:
        print(f"\n{'='*72}")
        print(f"Experiment: {exp_label}  ({folder})")
        print(f"{'='*72}")

        traces = load_regret(folder, n_runs, baselines=baselines)

        for key, label in TRACE_LABELS.items():
            if key not in traces:
                continue
            print(f"\n--- {label} ---")
            fit_trace(T_all, traces[key])
