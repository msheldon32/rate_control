import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_CSV = HERE / "preprocessed.csv"

LO, HI = 10 * 3600, 13 * 3600
EPSILON = 1
TIME_COLS = ["vru_entry", "vru_exit", "q_start", "q_exit", "ser_start", "ser_exit"]


def load_all():
    return pd.concat(
        [pd.read_csv(p, sep="\t", dtype=str) for p in sorted(DATA_DIR.glob("*1999.txt"))],
        ignore_index=True,
    )


def main():
    df = load_all()
    n0 = len(df)
    print(f"loaded {n0:,}")

    for c in TIME_COLS:
        df[c] = pd.to_timedelta(df[c].astype(str), errors="coerce").dt.total_seconds()
    df["q_time"] = pd.to_numeric(df["q_time"], errors="coerce")
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=TIME_COLS + ["q_time", "date"]).copy()
    for c in TIME_COLS:
        df[c] = df[c].astype(np.int64)
    df["date"] = df["date"].astype(np.int64)
    print(f"  parsed: {len(df):,}")

    dt = pd.to_datetime(df["date"].astype(str).str.zfill(6), format="%y%m%d")
    df = df[dt.dt.weekday.values < 4].copy()
    print(f"  Mon-Thu: {len(df):,}")

    df = df[(df["vru_entry"] >= LO) & (df["vru_entry"] < HI)].copy()
    print(f"  vru_entry in [{LO//3600}h,{HI//3600}h): {len(df):,}")

    df = df[df["type"] != "TT"].copy()
    print(f"  type != TT: {len(df):,}")

    df = df[df["outcome"] != "PHANTOM"].copy()
    print(f"  outcome != PHANTOM: {len(df):,}")

    df = df[df["q_time"] >= EPSILON].copy()
    print(f"  q_time >= {EPSILON}: {len(df):,}")

    counts = df.groupby("date").size()
    q1, q3 = counts.quantile(0.25), counts.quantile(0.75)
    keep = counts[(counts >= q1) & (counts <= q3)].index
    df = df[df["date"].isin(keep)].copy()
    print(f"  days w/ calls in IQR [{q1:.0f},{q3:.0f}]: {len(keep)} days, {len(df):,} calls")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nfinal: {len(df):,} ({len(df)/n0:.1%} of {n0:,}) → {OUT_CSV.name}")


if __name__ == "__main__":
    main()
