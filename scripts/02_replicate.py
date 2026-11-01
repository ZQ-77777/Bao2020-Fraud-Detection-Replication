"""Step 2: replicate the main table — RUSBoost (28 raw items) vs logit benchmarks, test years 2003-2008."""
import pandas as pd
from _common import parser, parse_years, out, C
from src import data as D, serial as S
from src.experiment import run, summarize

p = parser(__doc__)
p.add_argument("--skip-logit", action="store_true")
args = p.parse_args()
df = D.load(args.data)
if args.common_sample:
    df = df.dropna(subset=D.RAW28 + D.ratio_columns(df)).reset_index(drop=True)
    print("Common sample:", len(df), "rows")
years, seeds = parse_years(args.years), tuple(range(args.seeds))
ratios = D.ratio_columns(df)

specs = [("RUSBoost-raw28", "rusboost", D.RAW28)]
if not args.skip_logit:
    specs += [("Logit-raw28", "logit", D.RAW28)]
    if len(ratios) == 14:
        specs += [("Logit-ratio14 (Dechow et al. 2011 style)", "logit", ratios)]

schemes = ["recode_p_aaer"] + (["recode_new_p_aaer"] if "new_p_aaer" in df.columns else [])
allres = []
for scheme in schemes:
    for label, model, feats in specs:
        if model == "logit" and scheme != "recode_p_aaer":
            continue
        print(f"\n>>> {label} | serial-fraud scheme = {scheme}")
        allres.append(run(df, model, feats, scheme=scheme, years=years, seeds=seeds,
                          trees=args.trees, jobs=args.jobs, label=label))
res = pd.concat(allres, ignore_index=True)
res.to_csv(out("02_replicate_raw.csv"), index=False)
by_year, agg = summarize(res)
by_year.to_csv(out("02_replicate_by_year.csv"), index=False)
agg.to_csv(out("02_replicate_summary.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== Summary (means over test years; hits pooled over years) ===")
print(agg.round(4).to_string(index=False))
print("\nCheckpoints (Walker 2021, Table 2; top-1% hits, 2003-2008):")
print(f"  published model {C.CHECKPOINTS['hits_top1pct_published']} hits (PPV 4.5%), "
      f"re-run {C.CHECKPOINTS['hits_top1pct_rerun']}, recoded with p_aaer {C.CHECKPOINTS['hits_top1pct_recoded']} (PPV 2.5%); "
      f"logit benchmark PPV 2.63%")
