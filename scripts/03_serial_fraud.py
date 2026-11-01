"""Step 3: the serial-fraud debate — how much does the treatment of spanning frauds move the results?"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import parser, parse_years, out
from src import data as D, serial as S
from src.experiment import run, summarize

args = parser(__doc__).parse_args()
df = D.load(args.data)
years, seeds = parse_years(args.years), tuple(range(max(args.seeds, 1)))
order = ["none", "recode_new_p_aaer", "recode_p_aaer", "drop_p_aaer", "drop_firm"]
schemes = [s for s in order if s in S.available(df)]
res = pd.concat([run(df, "rusboost", D.RAW28, scheme=s, years=years, seeds=seeds, trees=args.trees, jobs=args.jobs, leaves=args.leaves,
                     label="RUSBoost-raw28") for s in schemes], ignore_index=True)
res.to_csv(out("03_serial_raw.csv"), index=False)
by_year, agg = summarize(res)
agg["scheme"] = pd.Categorical(agg["scheme"], schemes, ordered=True)
agg = agg.sort_values("scheme")
agg.to_csv(out("03_serial_summary.csv"), index=False)
adj = res.groupby(["scheme", "year"])["n_train_adjusted"].first().unstack()
adj.to_csv(out("03_rows_adjusted.csv"))
print("\n=== Serial-fraud schemes ===")
print(agg.round(4).to_string(index=False))
print("\nTraining fraud rows adjusted per test year:\n", adj.to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
names = [str(s) for s in agg["scheme"]]
axes[0].bar(names, agg["auc"]); axes[0].set_title("Mean AUC, 2003-2008"); axes[0].set_ylim(0.5, None)
axes[1].bar(names, agg["hits_total"]); axes[1].set_title("Frauds caught in top 1% (pooled)")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(out("03_serial_fraud.png"), dpi=160)
print("Saved results/03_*.")
