"""
Per-run log-log regression: log(regret) ~ alpha*log(T) + const (free intercept).

Fits each run independently, then reports mean slope (alpha), std, and 95% CI
via t-test across runs.  alpha ~ 0.5 => O(sqrt(T)), alpha ~ 1 => O(T).
"""

import pickle
import os
import numpy as np
from scipy import stats

TAIL_ONLY = True # if True, fit only the last 50% of time steps


def load_runs(folder, n_runs, baselines=False, normalize_regret=True, start_no=0):
    """Return dict of trace_name -> list of per-run regret arrays."""
    raw = {}
    ideal_gains = []

    for run_no in range(start_no, start_no + n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)

        ideal_gains.append(run_data["ideal_gain"])

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

    return raw


def fit_trace(T, runs):
    """Fit log(regret+1) ~ alpha*log(T+1), intercept fixed at 0, per run;
    aggregate slopes with t-test."""
    slopes = []
    for regret in runs:
        regret = np.array(regret)
        start = len(T) - (len(T) // 5) if TAIL_ONLY else 0
        T_fit = T[start:]
        r_fit = regret[start:]
        mask = (T_fit > 0) & (r_fit > 0)
        result = stats.linregress(np.log(T_fit[mask]), np.log(r_fit[mask]))
        slopes.append(result.slope)

    slopes = np.array(slopes)
    n = len(slopes)
    mean = slopes.mean()
    se = slopes.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.95, df=n - 1)
    ci_upper = mean + t_crit * se
    sublinear = ci_upper < 1.0

    print(f"  alpha:  mean={mean:.4f}  std={slopes.std(ddof=1):.4f}  "
          f"95% upper bound={ci_upper:.4f}  (n={n})  "
          f"{'** sublinear **' if sublinear else '(not significantly sublinear)'}   "
          f"min: {slopes.min():.4f}, max: {slopes.max():.4f}")


EXPERIMENTS = [
    ("exp_out/11_states/", 50, True, "11 states"),
    ("exp_out/21_states/", 50, True, "21 states"),
    ("exp_out/51_states/", 50, True, "51 states"),
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
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_label}  ({folder})")
        print(f"{'='*60}")

        runs = load_runs(folder, n_runs, baselines=baselines)

        for key, label in TRACE_LABELS.items():
            if key not in runs:
                continue
            print(f"  {label}")
            fit_trace(T_all, runs[key])
