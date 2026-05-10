"""
Fit regret(T) ~ a*f(T) + b for f in {T, sqrt(T), log(T)} across algorithm traces.
Reports slope, intercept, R², p-value, and 95% CI on slope for each fit.
"""

import pickle
import os
import numpy as np
from scipy import stats

# ── data loading (mirrors viz.py) ─────────────────────────────────────────────

def load_regret(folder, n_runs, baselines=False, normalize_regret=True, start_no=0):
    """Return dict of trace_name -> mean regret array (length = n_timesteps)."""
    raw = {}   # trace -> list of per-run regret arrays

    for run_no in range(start_no, start_no + n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)

        def normalize(l):
            if normalize_regret:
                return np.array(l) / run_data["ideal_gain"]
            return np.array(l)

        for key in ("rc", "ablation"):
            arr = normalize(run_data[key]["regret"])
            raw.setdefault(key, []).append(arr)

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
                arr = normalize(baseline_data[key]["regret"])
                raw.setdefault(key, []).append(arr)

    return {k: np.mean(v, axis=0) for k, v in raw.items()}


# ── regression helpers ────────────────────────────────────────────────────────

FITS = {
    "T":       lambda T: T,
    "sqrt(T)": lambda T: np.sqrt(T),
    "log(T)":  lambda T: np.log(T),
}

def fit_trace(T, regret, label):
    """Fit regret ~ a*f(T) + b for each basis; print a results table."""
    print(f"\n  {'Basis':<10}  {'slope':>14}  {'intercept':>14}  {'R²':>7}  {'p-value':>10}  {'95% CI slope'}")
    print(f"  {'-'*9}  {'-'*14}  {'-'*14}  {'-'*7}  {'-'*10}  {'-'*30}")

    best_r2 = -np.inf
    best_basis = None

    for basis_name, basis_fn in FITS.items():
        # drop T=0 for log, use all points otherwise
        mask = T > 0
        x = basis_fn(T[mask])
        y = regret[mask]

        result = stats.linregress(x, y)
        r2 = result.rvalue ** 2

        # 95% CI on slope: slope ± t_{0.975, n-2} * stderr
        n = mask.sum()
        t_crit = stats.t.ppf(0.975, df=n - 2)
        ci_lo = result.slope - t_crit * result.stderr
        ci_hi = result.slope + t_crit * result.stderr

        ci_str = f"[{ci_lo:+.4e}, {ci_hi:+.4e}]"
        print(f"  {basis_name:<10}  {result.slope:>+14.4e}  {result.intercept:>+14.4e}  {r2:>7.4f}  {result.pvalue:>10.3e}  {ci_str}")

        if r2 > best_r2:
            best_r2 = r2
            best_basis = basis_name

    print(f"  => best fit: {best_basis}  (R²={best_r2:.4f})")


# ── main ──────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("exp_out/11_states/",  50, True,  "11 states"),
    ("exp_out/21_states/",  50, True,  "21 states"),
    ("exp_out/51_states/",  50, True,  "51 states"),
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
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_label}  ({folder})")
        print(f"{'='*70}")

        traces = load_regret(folder, n_runs, baselines=baselines)

        for key, label in TRACE_LABELS.items():
            if key not in traces:
                continue
            print(f"\n--- {label} ---")
            fit_trace(T_all, traces[key], label)
