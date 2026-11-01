"""Rolling out-of-sample experiment: train 1991..(t-2), test on year t."""
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import config as C
from . import data as D, serial as S, models as M
from .metrics import evaluate


def _one(Xtr, ytr, Xte, yte, model, seed, trees, k, leaves=None):
    t0 = time.time()
    kw = {"trees": trees} if trees else {}
    if leaves is not None:
        kw["leaves"] = leaves
    mdl = M.make(model, seed=seed, **kw).fit(Xtr, ytr)
    s = mdl.score(Xte)
    r = {"seed": seed, "seconds": round(time.time() - t0, 1)}
    r.update(evaluate(yte, s, k))
    return r, s


def run(df, model, features, scheme="recode_p_aaer", years=None, seeds=(0,), trees=None,
        k=C.TOP_K, label=None, keep_scores=False, verbose=True, jobs=1, leaves=None):
    years = years or C.TEST_YEARS
    label = label or model
    tasks, meta = [], []
    for year in years:
        train, test = D.split(df, year, C.TRAIN_START, C.GAP)
        if len(test) == 0 or test["misstate"].sum() == 0:
            print(f"  [skip] test year {year}: {len(test)} rows, "
                  f"{int(test['misstate'].sum()) if len(test) else 0} frauds")
            continue
        train, n_adj = S.apply(train, test, scheme)
        train = train.dropna(subset=features)
        test = test.dropna(subset=features)
        Xtr, ytr = train[features].values, train["misstate"].values
        Xte, yte = test[features].values, test["misstate"].values
        gv = test["gvkey"].values if "gvkey" in test.columns else np.full(len(test), np.nan)
        for seed in seeds:
            tasks.append((Xtr, ytr, Xte, yte, model, seed, trees, k, leaves))
            meta.append({"label": label, "model": model, "scheme": scheme, "year": year,
                         "n_train": len(train), "n_train_fraud": int(ytr.sum()),
                         "n_train_adjusted": n_adj, "_gv": gv, "_y": yte})
    outs = Parallel(n_jobs=jobs)(delayed(_one)(*t) for t in tasks)
    rows, scores = [], {}
    for m, (r, s) in zip(meta, outs):
        row = {kk: v for kk, v in m.items() if not kk.startswith("_")}
        row.update(r)
        rows.append(row)
        if verbose:
            print(f"  [{label} | {scheme} | {row['year']} | seed {row['seed']}] AUC={row['auc']:.3f} "
                  f"NDCG@k={row['ndcg_at_k']:.3f} hits={row['hits']}/{row['n_top']} ({row['seconds']}s)", flush=True)
        scores.setdefault(row["year"], {"gv": m["_gv"], "y": m["_y"], "s": []})["s"].append(s)
    res = pd.DataFrame(rows)
    if not keep_scores:
        return res
    sc = pd.concat([pd.DataFrame({"year": yr, "gvkey": d["gv"], "misstate": d["y"],
                                  "score": np.mean(d["s"], axis=0), "label": label, "scheme": scheme})
                    for yr, d in scores.items()], ignore_index=True)
    return res, sc


def summarize(res):
    """Average over seeds within year, then report per-year means and pooled 2003-2008 hits."""
    cols = ["auc", "ndcg_at_k", "precision_at_k", "sensitivity_at_k", "hits", "n_top", "n_fraud"]
    by_year = res.groupby(["label", "scheme", "year"])[cols].mean().reset_index()
    agg = by_year.groupby(["label", "scheme"]).agg(
        auc=("auc", "mean"), ndcg_at_k=("ndcg_at_k", "mean"),
        sensitivity_at_k=("sensitivity_at_k", "mean"), precision_at_k=("precision_at_k", "mean"),
        hits_total=("hits", "sum"), top_total=("n_top", "sum")).reset_index()
    agg["ppv_pooled"] = agg["hits_total"] / agg["top_total"]
    tot = (res.groupby(["label", "scheme", "seed"])["hits"].sum()
              .groupby(["label", "scheme"]).agg(["min", "max"]))
    tot = tot.rename(columns={"min": "hits_min_seed", "max": "hits_max_seed"}).reset_index()
    return by_year, agg.merge(tot, on=["label", "scheme"])
