"""Step 1: check the data against the numbers reported in the paper and in Walker (2021)."""
import json
import pandas as pd
from _common import parser, out, C
from src import data as D, serial as S

args = parser(__doc__).parse_args()
df = D.load(args.data)
s = D.summarize(df)
print("\n=== Sample summary ===")
for k, v in s.items():
    exp = C.CHECKPOINTS.get(k)
    flag = "" if exp is None else ("  OK" if v == exp else f"  (expected {exp})")
    print(f"{k:28s} {v}{flag}")
print("Paper (Sec. 3.3): fraud duration mean / median / p90 = 2 / 2 / 4 years")

print("\nAAER id columns in this file:", D.id_columns(df))
print("Ratio columns found:", len(D.ratio_columns(df)), "of 14")
print("Serial-fraud schemes runnable with this file:", S.available(df))

by_year = df.groupby("fyear").agg(n=("misstate", "size"), fraud=("misstate", "sum"))
by_year["rate_pct"] = (100 * by_year["fraud"] / by_year["n"]).round(2)
print("\n=== Fraud firm-years by fiscal year ===")
print(by_year.to_string())
by_year.to_csv(out("01_fraud_by_year.csv"))

split = D.compare_id_versions(df)
if split is not None:
    print(f"\nAAERs split into several new_p_aaer ids: {len(split)} (Walker 2021 reports 17)")
    print(split.head(20).to_string(index=False))
    split.to_csv(out("01_split_aaer_ids.csv"), index=False)
else:
    print("\nThis file has no new_p_aaer column -> only the corrected (post-erratum) design can be run.")
    print("To reproduce the ORIGINAL published numbers, retrieve the older CSV from the repo history (see README).")

with open(out("01_summary.json"), "w") as f:
    json.dump(s, f, indent=2)
print("\nSaved results/01_*.")
