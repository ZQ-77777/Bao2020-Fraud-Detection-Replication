"""Step 6: which accounting items drive the fraud score? Permutation importance on a late test year."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import parser, out, C
from src import data as D, serial as S, models as M
from src.metrics import evaluate

p = parser(__doc__)
p.add_argument("--model", default="gbm", choices=["gbm", "rusboost", "logit"])
p.add_argument("--test-year", type=int, default=2008)
p.add_argument("--repeats", type=int, default=5)
args = p.parse_args()
df = D.load(args.data)
train, test = D.split(df, args.test_year, C.TRAIN_START, C.GAP)
train, _ = S.apply(train, test, "recode_p_aaer")
kw = {"trees": args.trees} if args.trees else {}
mdl = M.make(args.model, seed=0, **kw).fit(train[D.RAW28].values, train.misstate.values)
X, y = test[D.RAW28].values.copy(), test.misstate.values
base = evaluate(y, mdl.score(X))["auc"]
rng = np.random.default_rng(0)
rows = []
for j, c in enumerate(D.RAW28):
    drops = []
    for _ in range(args.repeats):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base - evaluate(y, mdl.score(Xp))["auc"])
    rows.append({"variable": c, "auc_drop_mean": np.mean(drops), "auc_drop_sd": np.std(drops)})
imp = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)
imp.to_csv(out(f"06_importance_{args.model}_{args.test_year}.csv"), index=False)
print(f"Baseline AUC {base:.4f} ({args.model}, test {args.test_year})\n", imp.round(4).to_string(index=False))
top = imp.head(12)[::-1]
plt.figure(figsize=(5.5, 4)); plt.barh(top.variable, top.auc_drop_mean, xerr=top.auc_drop_sd)
plt.xlabel("AUC drop when permuted"); plt.title(f"Permutation importance ({args.model}, {args.test_year})")
plt.tight_layout(); plt.savefig(out(f"06_importance_{args.model}_{args.test_year}.png"), dpi=160)
