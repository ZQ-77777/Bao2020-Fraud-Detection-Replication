"""Step 5: replicate the authors' tuning — number of trees chosen by AUC, train 1991-1999, validate 2001."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import parser, out, C
from src import data as D, models as M, serial as S
from src.metrics import evaluate

args = parser(__doc__).parse_args()
df = D.load(args.data)
train = df[(df.fyear >= C.TRAIN_START) & (df.fyear <= C.TUNE_TRAIN_END)]
valid = df[df.fyear == C.TUNE_VALID_YEAR]
scheme = "recode_p_aaer" if "p_aaer" in df.columns else "none"
train, n_adj = S.apply(train, valid, scheme)
print(f"Serial-fraud handling in tuning ({scheme}): {n_adj} training fraud rows recoded")
cap = args.trees or max(C.TUNE_GRID)
grid = sorted(set([g for g in C.TUNE_GRID if g <= cap] + [cap]))
rows = []
for seed in range(max(args.seeds, 1)):
    kw = {"leaves": args.leaves} if args.leaves is not None else {}
    mdl = M.make("rusboost", seed=seed, trees=max(grid), **kw).fit(train[D.RAW28].values, train.misstate.values)
    staged = mdl.staged_scores(valid[D.RAW28].values, grid)
    for n, s in staged.items():
        r = {"seed": seed, "n_trees": n}; r.update(evaluate(valid.misstate.values, s, C.TOP_K)); rows.append(r)
        print(f"seed {seed} trees {n:5d} AUC={r['auc']:.4f}")
res = pd.DataFrame(rows)
res.to_csv(out("05_tuning.csv"), index=False)
m = res.groupby("n_trees")["auc"].mean()
print("\nBest number of trees by mean validation AUC:", int(m.idxmax()))
plt.figure(figsize=(5, 3.4)); plt.plot(m.index, m.values, marker="o")
plt.xlabel("number of trees"); plt.ylabel("validation AUC (2001)"); plt.tight_layout()
plt.savefig(out("05_tuning.png"), dpi=160)
