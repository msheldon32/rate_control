import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
IN_CSV = HERE / "preprocessed.csv"
OUT_FIG = HERE / "hourly_stats.pdf"
OUT_FIG_STATE = HERE / "state_stats.pdf"

HOURS = list(range(11, 14))
DAY_LO, DAY_HI = HOURS[0] * 3600, (HOURS[-1] + 1) * 3600
EPOCH = DAY_LO
MAX_S = 100


def hour_of(t):
    return ((t.astype(np.int64) % 86400) // 3600).astype(int)


def arrival_rate_by_hour(df):
    h = hour_of(df["q_start"].values)
    n_days = df["date"].nunique()
    return {hh: (h == hh).sum() / (n_days * 3600) for hh in HOURS}


def service_rate_by_hour(df):
    served = df[df["outcome"] == "AGENT"].copy()
    served["dur"] = served["ser_exit"] - served["ser_start"]
    served = served[served["dur"] > 0]
    served["hour"] = hour_of(served["ser_start"].values)
    out = {}
    for hh in HOURS:
        x = served.loc[served["hour"] == hh, "dur"].values.astype(float)
        out[hh] = 1.0 / x.mean() if x.size > 0 and x.mean() > 0 else np.nan
    return out


def scv_interarrival_by_hour(df):
    df = df.sort_values(["date", "q_start"]).copy()
    df["hour"] = hour_of(df["q_start"].values)
    by_hour = {hh: [] for hh in HOURS}
    for _, day_sub in df.groupby("date", sort=False):
        t = day_sub["q_start"].values.astype(float)
        h = day_sub["hour"].values
        prev = float(EPOCH)
        for i in range(len(t)):
            if h[i] in by_hour:
                by_hour[h[i]].append(t[i] - prev)
            prev = t[i]
    out = {}
    for hh in HOURS:
        x = np.array(by_hour[hh], dtype=float)
        out[hh] = x.var() / x.mean() ** 2 if (x.size > 1 and x.mean() > 0) else np.nan
    return out


def scv_service_by_hour(df):
    served = df[df["outcome"] == "AGENT"].copy()
    served["dur"] = served["ser_exit"] - served["ser_start"]
    served = served[served["dur"] > 0]
    served["hour"] = hour_of(served["ser_start"].values)
    out = {}
    for hh in HOURS:
        x = served.loc[served["hour"] == hh, "dur"].values.astype(float)
        out[hh] = x.var() / x.mean() ** 2 if (len(x) > 1 and x.mean() > 0) else np.nan
    return out


def state_dwell_by_hour(df, max_s=MAX_S):
    dwell = np.zeros((24, max_s))
    served = (df["outcome"] == "AGENT").values
    arr_t = df["q_start"].values.astype(float)
    dep_t = np.where(served, df["ser_exit"].values, df["q_exit"].values).astype(float)
    work = pd.DataFrame({"date": df["date"].values, "t_arr": arr_t, "t_dep": dep_t})

    def add(t0, t1, state):
        while t0 < t1:
            h = int(t0 // 3600)
            nb = (h + 1) * 3600
            te = min(t1, nb)
            if 0 <= state < max_s and 0 <= h < 24:
                dwell[h, state] += te - t0
            t0 = te

    for _, sub in work.groupby("date"):
        a = sub["t_arr"].values
        d = sub["t_dep"].values
        n = len(a) + len(d)
        times = np.empty(n)
        deltas = np.empty(n, dtype=np.int8)
        times[: len(a)] = a
        deltas[: len(a)] = 1
        times[len(a) :] = d
        deltas[len(a) :] = -1
        order = np.argsort(times, kind="stable")
        times = times[order]
        deltas = deltas[order]

        state = 0
        last_t = float(DAY_LO)
        capped = False
        for t, dlt in zip(times, deltas):
            add(last_t, min(t, DAY_HI), state)
            last_t = t
            if last_t >= DAY_HI:
                capped = True
                break
            state += dlt
        if not capped:
            add(last_t, DAY_HI, state)
    return dwell


def state_stats_by_hour(dwell):
    s = np.arange(dwell.shape[1])
    mean_L, scv_L = {}, {}
    for hh in HOURS:
        d = dwell[hh]
        tot = d.sum()
        if tot == 0:
            mean_L[hh] = scv_L[hh] = np.nan
            continue
        p = d / tot
        m = float((s * p).sum())
        m2 = float((s ** 2 * p).sum())
        v = m2 - m ** 2
        mean_L[hh] = m
        scv_L[hh] = v / m ** 2 if m > 0 else np.nan
    return mean_L, scv_L


def state_metrics(df, max_s=MAX_S):
    """Single pass per day. Tracks dwell and event counts per state, plus
    interarrival times bucketed by state at new arrival, and service durations
    bucketed by state at ser_start."""
    dwell = np.zeros(max_s)
    n_arr = np.zeros(max_s, dtype=np.int64)
    n_dep = np.zeros(max_s, dtype=np.int64)
    iats = [[] for _ in range(max_s)]
    svcs = [[] for _ in range(max_s)]

    df = df.sort_values(["date", "q_start"])
    for _, sub in df.groupby("date", sort=False):
        served = (sub["outcome"] == "AGENT").values
        a = sub["q_start"].values.astype(float)
        d = np.where(served, sub["ser_exit"].values, sub["q_exit"].values).astype(float)
        sd = np.where(
            served,
            sub["ser_exit"].values - sub["ser_start"].values,
            0.0,
        ).astype(float)

        n = 2 * len(sub)
        times = np.empty(n)
        kinds = np.empty(n, dtype=np.int8)
        payloads = np.zeros(n)
        m = len(sub)
        times[:m] = a; kinds[:m] = 0
        times[m:] = d; kinds[m:] = 1; payloads[m:] = sd

        order = np.argsort(times, kind="stable")
        times = times[order]; kinds = kinds[order]; payloads = payloads[order]

        state = 0
        last_t = float(DAY_LO)
        prev_arr = float(EPOCH)
        for j in range(len(times)):
            t = times[j]
            t_clip = min(t, DAY_HI)
            if t_clip > last_t and 0 <= state < max_s:
                dwell[state] += t_clip - last_t
            last_t = t_clip
            if t > DAY_HI:
                break
            k = kinds[j]
            if 0 <= state < max_s:
                if k == 0:
                    n_arr[state] += 1
                    iats[state].append(t - prev_arr)
                    prev_arr = t
                else:
                    n_dep[state] += 1
                    if payloads[j] > 0:
                        svcs[state].append(payloads[j])
            if k == 0:
                state += 1
            else:
                state -= 1
        if last_t < DAY_HI and 0 <= state < max_s:
            dwell[state] += DAY_HI - last_t

    return dwell, n_arr, n_dep, iats, svcs


def state_summary(dwell, n_arr, n_dep, iats, svcs):
    max_s = len(dwell)
    lam = np.full(max_s, np.nan)
    mu_ag = np.full(max_s, np.nan)
    mu_pc = np.full(max_s, np.nan)
    scv_iat = np.full(max_s, np.nan)
    scv_svc = np.full(max_s, np.nan)

    ok = dwell > 0
    lam[ok] = n_arr[ok] / dwell[ok]
    mu_ag[ok] = n_dep[ok] / dwell[ok]

    for s in range(max_s):
        x = np.array(iats[s], dtype=float)
        if x.size > 1 and x.mean() > 0:
            scv_iat[s] = x.var() / x.mean() ** 2
        y = np.array(svcs[s], dtype=float)
        if y.size > 1 and y.mean() > 0:
            scv_svc[s] = y.var() / y.mean() ** 2
            mu_pc[s] = 1.0 / y.mean()
        elif y.size == 1 and y[0] > 0:
            mu_pc[s] = 1.0 / y[0]
    return lam, mu_ag, mu_pc, scv_iat, scv_svc


def plot_state(dwell, lam, mu_ag, mu_pc, scv_iat, scv_svc, path):
    tot = dwell.sum()
    smax = max(15, int(np.searchsorted(np.cumsum(dwell) / max(tot, 1), 0.99)) + 1)
    s = np.arange(smax)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)

    axes[0, 0].bar(s, lam[:smax], width=0.85, alpha=0.85, color="C0")
    axes[0, 0].set_ylabel("arrivals / sec")
    axes[0, 0].set_title("arrival rate λ̂(s)")

    axes[0, 1].bar(s, mu_pc[:smax], width=0.85, alpha=0.85, color="C3")
    axes[0, 1].plot(s, mu_ag[:smax], "o-", color="k", ms=3, lw=1, label="aggregate n_dep/dwell")
    axes[0, 1].set_ylabel("rate [1/s]")
    axes[0, 1].set_title("service rate μ̂(s) = 1/E[ser_exit − ser_start | s]")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(s, scv_iat[:smax], "o-", color="C1")
    axes[1, 0].axhline(1, ls="--", color="k", alpha=0.4, label="Poisson")
    axes[1, 0].set_ylabel("SCV")
    axes[1, 0].set_title("interarrival SCV by state")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(s, scv_svc[:smax], "o-", color="C2")
    axes[1, 1].axhline(1, ls="--", color="k", alpha=0.4, label="exp")
    axes[1, 1].set_ylabel("SCV")
    axes[1, 1].set_title("service-time SCV by state")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].bar(s, dwell[:smax], width=0.85, alpha=0.85, color="C4")
    axes[2, 0].set_ylabel("dwell [s]")
    axes[2, 0].set_xlabel("state s")
    axes[2, 0].set_title("dwell time at state s")

    p = dwell[:smax] / max(tot, 1)
    axes[2, 1].bar(s, p, width=0.85, alpha=0.85, color="C5")
    axes[2, 1].set_ylabel("P(L=s)")
    axes[2, 1].set_xlabel("state s")
    axes[2, 1].set_title("state occupancy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    return smax


def plot(arr, mu, scv_a, scv_s, mean_L, scv_L, path):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)

    axes[0, 0].bar(HOURS, [arr[h] for h in HOURS], width=0.85, alpha=0.85, color="C0")
    axes[0, 0].set_ylabel("arrivals / sec")
    axes[0, 0].set_title("arrival rate λ̂")

    axes[0, 1].bar(HOURS, [mu[h] for h in HOURS], width=0.85, alpha=0.85, color="C3")
    axes[0, 1].set_ylabel("1/E[svc]  [1/s]")
    axes[0, 1].set_title("per-call service rate μ̂")

    axes[1, 0].plot(HOURS, [scv_a[h] for h in HOURS], "o-", color="C1")
    axes[1, 0].axhline(1, ls="--", color="k", alpha=0.4, label="Poisson")
    axes[1, 0].set_ylabel("SCV")
    axes[1, 0].set_title("interarrival SCV")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(HOURS, [scv_s[h] for h in HOURS], "o-", color="C2")
    axes[1, 1].axhline(1, ls="--", color="k", alpha=0.4, label="exp")
    axes[1, 1].set_ylabel("SCV")
    axes[1, 1].set_title("service-time SCV")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].bar(HOURS, [mean_L[h] for h in HOURS], width=0.85, alpha=0.85, color="C4")
    axes[2, 0].set_ylabel("E[L]")
    axes[2, 0].set_xlabel("hour")
    axes[2, 0].set_title("mean # in system")

    axes[2, 1].plot(HOURS, [scv_L[h] for h in HOURS], "o-", color="C5")
    axes[2, 1].set_ylabel("SCV")
    axes[2, 1].set_xlabel("hour")
    axes[2, 1].set_title("SCV of # in system")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.set_xticks(HOURS)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)


def main():
    df = pd.read_csv(IN_CSV).sort_values(["date", "q_start"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows, {df['date'].nunique()} days")
    arr = arrival_rate_by_hour(df)
    mu = service_rate_by_hour(df)
    scv_a = scv_interarrival_by_hour(df)
    scv_s = scv_service_by_hour(df)
    dwell = state_dwell_by_hour(df)
    mean_L, scv_L = state_stats_by_hour(dwell)

    plot(arr, mu, scv_a, scv_s, mean_L, scv_L, OUT_FIG)
    print(f"figure → {OUT_FIG.name}\n")
    print("hour    λ̂        μ̂       SCV(IA)  SCV(svc)   E[L]    SCV(L)")
    for hh in HOURS:
        print(
            f"  {hh:>2}   {arr[hh]:.4f}   {mu[hh]:.4f}    {scv_a[hh]:>5.2f}    {scv_s[hh]:>5.2f}    "
            f"{mean_L[hh]:>5.2f}    {scv_L[hh]:>5.2f}"
        )

    s_dwell, s_arr, s_dep, s_iats, s_svcs = state_metrics(df)
    lam_s, mu_ag, mu_pc, scv_iat_s, scv_svc_s = state_summary(s_dwell, s_arr, s_dep, s_iats, s_svcs)
    smax = plot_state(s_dwell, lam_s, mu_ag, mu_pc, scv_iat_s, scv_svc_s, OUT_FIG_STATE)
    print(f"\nfigure → {OUT_FIG_STATE.name}\n")
    print("state   dwell    λ̂(s)     μ̂(s)     1/E[svc]  SCV(IA)  SCV(svc)   #IAT   #svc")
    for s in range(smax):
        print(
            f"  {s:>3}  {s_dwell[s]:>8.0f}  {lam_s[s]:.4f}   {mu_ag[s]:.4f}   {mu_pc[s]:.4f}    "
            f"{scv_iat_s[s]:>5.2f}    {scv_svc_s[s]:>5.2f}   {len(s_iats[s]):>5}  {len(s_svcs[s]):>5}"
        )


if __name__ == "__main__":
    main()
