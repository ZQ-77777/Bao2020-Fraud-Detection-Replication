"""Create a SYNTHETIC file with the same columns as the real data, only to test the pipeline.
Never use its numbers for anything else."""
import numpy as np
import pandas as pd
from _common import ROOT
import sys
sys.path.insert(0, str(ROOT))
from src.data import RAW28, RATIO14

rng = np.random.default_rng(42)
rows = []
gv = 1000
firms = [gv + i for i in range(2500)]
aaer = 1000
for f in firms:
    start = rng.integers(1991, 2010)
    length = rng.integers(3, 12)
    fraud_years, pid = set(), 0
    if rng.random() < 0.06:
        fs = rng.integers(start, start + length)
        dur = rng.integers(1, 5)
        fraud_years = set(range(fs, fs + dur))
        aaer += 1
        pid = aaer
    size = rng.lognormal(5, 1.5)
    for y in range(start, min(start + length, 2015)):
        fr = int(y in fraud_years)
        r = {"fyear": y, "gvkey": f, "sich": 2000, "insbnk": 0, "understatement": 0, "option": 0,
             "p_aaer": pid if fr else 0, "new_p_aaer": (pid * 10 + 1) if fr else 0, "misstate": fr}
        base = size * rng.lognormal(0, 0.2)
        for c in RAW28:
            r[c] = base * rng.lognormal(0, 0.5)
        if fr:  # make frauds weakly detectable
            r["rect"] *= 1.8
            r["invt"] *= 1.5
            r["sstk"] *= 2.0
        for c in RATIO14:
            r[c] = rng.normal(0.3 * fr, 1)
        rows.append(r)
df = pd.DataFrame(rows)
# mimic the 'split id' issue for a few AAERs: second half of the spell gets suffix 2
for pid in df.loc[df.misstate == 1, "p_aaer"].unique()[:15]:
    idx = df.index[(df.p_aaer == pid)]
    if len(idx) > 1:
        df.loc[idx[len(idx) // 2:], "new_p_aaer"] = pid * 10 + 2
out = ROOT / "data" / "synthetic_test.csv"
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, {df.misstate.sum()} fraud firm-years (SYNTHETIC)")
