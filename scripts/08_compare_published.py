"""Step 8: compare with the authors' published RUSBoost predictions (2003-2008).

Run on the OLD data file so the row order matches the authors' test sets:
    python scripts/08_compare_published.py --data data/uscecchini28.csv
    python scripts/08_compare_published.py --data data/uscecchini28.csv --ours   # also fit our model and compare scores
"""
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from _common import parser, out, C
from src import data as D, serial as S, models as M
from src.metrics import evaluate

p = parser(__doc__)
p.add_argument("--pred-dir", default=str(C.DATA_DIR / "published_predictions"))
p.add_argument("--ours", action="store_true", help="fit our RUSBoost with the authors' original design and compare")
args = p.parse_args()
df = D.load(args.data)
scheme = "recode_new_p_aaer" if "new_p_aaer" in df.columns else "recode_p_aaer"


def read_pred(f):
    raw = pd.read_csv(f, header=None)
    if not np.issubdtype(raw.dtypes.iloc[0], np.number):        # first row is a header
        raw = pd.read_csv(f)
    num = raw.apply(pd.to_numeric, errors="coerce")
    binary = [c for c in num.columns if set(num[c].dropna().unique()) <= {0, 1}]
    cont = [c for c in num.columns if c not in binary]
    return raw, num, binary, cont


files = sorted(glob.glob(f"{args.pred_dir}/prediction_rusboost28_*.csv"))
if not files:
    raise SystemExit(f"No prediction files found in {args.pred_dir}")
rows = []
for i, f in enumerate(files):
    year = int(re.findall(r"(\d{4})\.csv$", f)[0])
    raw, num, binary, cont = read_pred(f)
    if i == 0:
        print("File format (first rows of", f.split("\\")[-1].split("/")[-1], "):")
        print(raw.head().to_string(), "\nbinary cols:", binary, "| continuous cols:", cont, "\n")
    test = df[df["fyear"] == year]
    r = {"year": year, "n_pred_file": len(raw), "n_test_data": len(test)}
    # label column = binary column whose total equals the number of frauds in the test year (if any)
    lab = next((c for c in binary if int(num[c].sum()) == int(test["misstate"].sum())), binary[0] if binary else None)
    score = cont[-1] if cont else None
    if lab is None or score is None:
        print(f"{year}: could not identify label/score columns; columns = {list(raw.columns)}")
        continue
    pub = evaluate(num[lab].values, num[score].values, C.TOP_K)
    r.update({f"pub_{k}": v for k, v in pub.items() if k in ["auc", "ndcg_at_k", "hits", "n_top", "n_fraud"]})
    aligned = len(raw) == len(test) and np.array_equal(num[lab].values.astype(int), test["misstate"].values)
    r["aligned_with_data"] = aligned
    if args.ours and aligned:
        train, _ = D.split(df, year, C.TRAIN_START, C.GAP)
        train, _ = S.apply(train, test, scheme)
        kw = {"trees": args.trees} if args.trees else {}
        ours = M.make("rusboost", seed=0, **kw).fit(train[D.RAW28].values, train["misstate"].values)
        s = ours.score(test[D.RAW28].values)
        mine = evaluate(test["misstate"].values, s, C.TOP_K)
        k = mine["n_top"]
        top_pub = set(np.argsort(-num[score].values)[:k])
        top_me = set(np.argsort(-s)[:k])
        r.update({"our_auc": mine["auc"], "our_hits": mine["hits"],
                  "spearman_scores": float(spearmanr(s, num[score].values).correlation),
                  "top1pct_overlap": len(top_pub & top_me) / k})
    rows.append(r)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
res = pd.DataFrame(rows)
res.to_csv(out("08_published_compare.csv"), index=False)
print("\nPublished predictions, 2003-2008: total hits in top 1% =", int(res["pub_hits"].sum()),
      f"(paper / Walker: {C.CHECKPOINTS['hits_top1pct_published']}); mean AUC = {res['pub_auc'].mean():.4f}")
if "our_hits" in res:
    print(f"Ours ({scheme}): total hits = {int(res['our_hits'].sum())}, mean AUC = {res['our_auc'].mean():.4f}, "
          f"mean Spearman with published = {res['spearman_scores'].mean():.3f}, "
          f"mean top-1% overlap = {res['top1pct_overlap'].mean():.2f}")
