"""Step 10: which kind of test-year fraud does each model catch?

Every fraud firm-year in test year t is put in one category, based on what the training window
(1991..t-2) contains for the same firm:
  split_leak : same AAER (p_aaer) has training fraud years that the authors' ORIGINAL code did NOT
               recode, because new_p_aaer split the case into two ids  -> leaked in the published design
  same_case  : same AAER has training fraud years, recoded in both the original and corrected design
  other_case : the firm has training fraud years from a DIFFERENT AAER (never recoded in any design)
  new        : the firm has no fraud in the training window
Needs the OLD file (with new_p_aaer):
    python scripts/10_leak_categories.py --data data/uscecchini28.csv
    python scripts/10_leak_categories.py --data data/uscecchini28.csv --ours --seeds 3
"""
import glob
import re
import numpy as np
import pandas as pd
from _common import parser, out, C
from src import data as D, serial as S, models as M

p = parser(__doc__)
p.add_argument("--pred-dir", default=str(C.DATA_DIR / "published_predictions"))
p.add_argument("--ours", action="store_true")
p.add_argument("--schemes", default="recode_new_p_aaer,recode_p_aaer")
args = p.parse_args()
df = D.load(args.data)
need = {"gvkey", "p_aaer", "new_p_aaer"}
if not need.issubset(df.columns):
    raise SystemExit(f"Needs columns {need}; use data/uscecchini28.csv")
df["pid"] = D.clean_ids(df["p_aaer"])
df["npid"] = D.clean_ids(df["new_p_aaer"])
CATS = ["split_leak", "same_case", "other_case", "new"]


def categories(year):
    test = df[df["fyear"] == year]
    tr = df[(df["fyear"] >= C.TRAIN_START) & (df["fyear"] <= year - C.GAP) & (df["misstate"] == 1)]
    test_new_ids = set(test.loc[test["misstate"] == 1, "npid"].dropna())
    cat = np.array([""] * len(test), dtype=object)
    for i, (m, gv, pid) in enumerate(zip(test["misstate"], test["gvkey"], test["pid"])):
        if m != 1:
            continue
        same = tr[tr["pid"] == pid] if pd.notna(pid) else tr.iloc[0:0]
        if len(same):
            cat[i] = "split_leak" if (~same["npid"].isin(test_new_ids)).any() else "same_case"
        elif gv in set(tr["gvkey"]):
            cat[i] = "other_case"
        else:
            cat[i] = "new"
    return test, cat


def count(source, year, y, score, cat, seed=None):
    n_top = max(1, int(round(C.TOP_K * len(y))))
    top = np.argsort(-score, kind="mergesort")[:n_top]
    r = {"source": source, "year": year, "seed": seed, "hits": int(y[top].sum())}
    for c in CATS:
        r[c] = int(((cat[top] == c) & (y[top] == 1)).sum())
    return r


files = {int(re.findall(r"(\d{4})\.csv$", f)[0]): f for f in glob.glob(f"{args.pred_dir}/prediction_rusboost28_*.csv")}
schemes = [s for s in args.schemes.split(",") if s in S.available(df)] if args.ours else []
rows, base = [], []
for year in C.TEST_YEARS:
    test, cat = categories(year)
    y = test["misstate"].values
    b = {"source": "all test-year frauds", "year": year, "hits": int(y.sum())}
    b.update({c: int((cat == c).sum()) for c in CATS})
    base.append(b)
    if year in files:
        pub = pd.read_csv(files[year], header=None).apply(pd.to_numeric, errors="coerce")
        lab = next(c for c in pub.columns if set(pub[c].dropna().unique()) <= {0, 1})
        if len(pub) == len(test) and np.array_equal(pub[lab].values.astype(int), y):
            rows.append(count("published", year, y, pub[pub.columns[-1]].values, cat))
        else:
            print(f"{year}: published file not aligned, skipped")
    for scheme in schemes:
        train, _ = D.split(df, year, C.TRAIN_START, C.GAP)
        train, _ = S.apply(train, test, scheme)
        kw = {"trees": args.trees} if args.trees else {}
        if args.leaves is not None:
            kw["leaves"] = args.leaves
        for seed in range(args.seeds):
            mdl = M.make("rusboost", seed=seed, **kw).fit(train[D.RAW28].values, train["misstate"].values)
            rows.append(count(f"ours_{scheme}", year, y, mdl.score(test[D.RAW28].values), cat, seed))
            print(f"  {year} {scheme} seed {seed} done", flush=True)

res = pd.concat([pd.DataFrame(base), pd.DataFrame(rows)], ignore_index=True)
tag = "" if args.leaves is None else f"_leaves{args.leaves}"
res.to_csv(out(f"10_leak_categories{tag}.csv"), index=False)
summ = (res.groupby(["source", "year"])[["hits"] + CATS].mean().groupby("source").sum())
order = ["all test-year frauds", "published"] + [f"ours_{s}" for s in schemes]
summ = summ.reindex([o for o in order if o in summ.index])
summ.to_csv(out(f"10_leak_categories_summary{tag}.csv"))
pd.set_option("display.width", 200)
print("\n=== Top-1% hits by fraud category, 2003-2008 totals (ours = mean over seeds) ===")
print(summ.round(2).to_string())
print("\nRead: if the published model's extra hits sit in split_leak, Walker's mechanism is confirmed firm by firm.")
