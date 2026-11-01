"""Step 9: are the frauds caught in the top 1% mostly 'serial' frauds that a model could have memorized?

A test-year fraud is called SERIAL if the same firm (gvkey) already has fraud firm-years inside the
training window (1991..t-2). If the published model's hits are far more serial than the pool of all
test-year frauds, its edge comes from recognizing firms rather than detecting fraud.

    python scripts/09_memorization_check.py --data data/uscecchini28.csv
    python scripts/09_memorization_check.py --data data/uscecchini28.csv --ours --leaves 0
"""
import glob
import re
import numpy as np
import pandas as pd
from _common import parser, out, C
from src import data as D, serial as S, models as M

p = parser(__doc__)
p.add_argument("--pred-dir", default=str(C.DATA_DIR / "published_predictions"))
p.add_argument("--ours", action="store_true", help="also fit our RUSBoost under the schemes below")
p.add_argument("--schemes", default="recode_new_p_aaer,recode_p_aaer,drop_firm")
args = p.parse_args()
df = D.load(args.data)
if "gvkey" not in df.columns:
    raise SystemExit("This check needs a gvkey column.")


def breakdown(y, score, serial, k=C.TOP_K):
    n_top = max(1, int(round(k * len(y))))
    top = np.argsort(-score, kind="mergesort")[:n_top]
    hit = y[top] == 1
    return {"n_top": n_top, "hits": int(hit.sum()), "hits_serial": int((hit & serial[top]).sum()),
            "top_serial_firms": int(serial[top].sum())}


rows = []
files = {int(re.findall(r"(\d{4})\.csv$", f)[0]): f
         for f in glob.glob(f"{args.pred_dir}/prediction_rusboost28_*.csv")}
schemes = [s for s in args.schemes.split(",") if s in S.available(df)] if args.ours else []
for year in C.TEST_YEARS:
    test = df[df["fyear"] == year]
    prior = df[(df["fyear"] >= C.TRAIN_START) & (df["fyear"] <= year - C.GAP) & (df["misstate"] == 1)]
    serial = test["gvkey"].isin(set(prior["gvkey"])).values
    y = test["misstate"].values
    base = {"year": year, "n_fraud": int(y.sum()), "n_fraud_serial": int((y == 1)[serial].sum())}
    if year in files:
        pub = pd.read_csv(files[year], header=None).apply(pd.to_numeric, errors="coerce")
        lab_col = next(c for c in pub.columns if set(pub[c].dropna().unique()) <= {0, 1})
        ok = len(pub) == len(test) and np.array_equal(pub[lab_col].values.astype(int), y)
        if not ok:
            print(f"{year}: published file not aligned with data, skipped")
        else:
            r = dict(base, source="published")
            r.update(breakdown(y, pub[pub.columns[-1]].values, serial))
            rows.append(r)
    for scheme in schemes:
        train, _ = D.split(df, year, C.TRAIN_START, C.GAP)
        train, _ = S.apply(train, test, scheme)
        kw = {"trees": args.trees} if args.trees else {}
        if args.leaves is not None:
            kw["leaves"] = args.leaves
        for seed in range(args.seeds):
            mdl = M.make("rusboost", seed=seed, **kw).fit(train[D.RAW28].values, train["misstate"].values)
            r = dict(base, source=f"ours_{scheme}", seed=seed)
            r.update(breakdown(y, mdl.score(test[D.RAW28].values), serial))
            rows.append(r)
            print(f"  {year} {scheme} seed {seed}: hits {r['hits']} (serial {r['hits_serial']})", flush=True)

res = pd.DataFrame(rows)
tag = "" if args.leaves is None else f"_leaves{args.leaves}"
res.to_csv(out(f"09_memorization{tag}.csv"), index=False)
per_source = (res.groupby(["source", "year"]).mean(numeric_only=True).groupby("source")
                 [["n_fraud", "n_fraud_serial", "n_top", "hits", "hits_serial", "top_serial_firms"]].sum())
per_source["share_serial_among_test_frauds"] = per_source["n_fraud_serial"] / per_source["n_fraud"]
per_source["share_serial_among_hits"] = per_source["hits_serial"] / per_source["hits"].where(per_source["hits"] > 0)
per_source["hits_new_frauds"] = per_source["hits"] - per_source["hits_serial"]
per_source.to_csv(out(f"09_memorization_summary{tag}.csv"))
pd.set_option("display.width", 200)
print("\n=== Top-1% hits split into serial vs new frauds (2003-2008 totals; ours = mean over seeds) ===")
print(per_source.round(3).to_string())
print("\nRead: if share_serial_among_hits >> share_serial_among_test_frauds, the model mostly re-finds firms it saw as frauds.")
