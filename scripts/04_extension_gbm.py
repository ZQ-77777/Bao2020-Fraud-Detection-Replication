"""Step 4: extension — does a modern gradient-boosting model (as in Geertsema & Lu) beat RUSBoost,
and do the 14 theory-based ratios add information beyond the 28 raw items?"""
import pandas as pd
from _common import parser, parse_years, out
from src import data as D
from src.experiment import run, summarize

args = parser(__doc__).parse_args()
df = D.load(args.data)
if args.common_sample:
    df = df.dropna(subset=D.RAW28 + D.ratio_columns(df)).reset_index(drop=True)
    print("Common sample:", len(df), "rows")
years, seeds = parse_years(args.years), tuple(range(args.seeds))
ratios = D.ratio_columns(df)
specs = [("RUSBoost-raw28", "rusboost", D.RAW28),
         ("GBM-raw28", "gbm", D.RAW28)]
if len(ratios) == 14:
    specs += [("RUSBoost-raw28+ratio14", "rusboost", D.RAW28 + ratios),
              ("GBM-raw28+ratio14", "gbm", D.RAW28 + ratios),
              ("GBM-ratio14", "gbm", ratios)]
res, scores = [], []
for label, model, feats in specs:
    print(f"\n>>> {label}")
    r, s = run(df, model, feats, scheme="recode_p_aaer", years=years, seeds=seeds, trees=args.trees, jobs=args.jobs,
               label=label, keep_scores=True)
    res.append(r); scores.append(s)
res = pd.concat(res, ignore_index=True)
pd.concat(scores, ignore_index=True).to_csv(out("04_scores.csv"), index=False)
res.to_csv(out("04_extension_raw.csv"), index=False)
by_year, agg = summarize(res)
by_year.to_csv(out("04_extension_by_year.csv"), index=False)
agg.to_csv(out("04_extension_summary.csv"), index=False)
print("\n=== Extension summary (corrected serial-fraud design) ===")
print(agg.round(4).to_string(index=False))
