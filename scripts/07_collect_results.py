"""Step 7: collect all summaries into results/REPORT.md (paste this back to Claude for interpretation)."""
import json
import pandas as pd
from _common import out, C


def md(df, digits=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(digits)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join([head, sep] + body)


parts = ["# Replication results: Bao et al. (2020, JAR)\n"]
f = out("01_summary.json")
if f.exists():
    s = json.load(open(f))
    parts.append("## 1. Data check\n")
    rows = [(k, (f"{v:.4f}" if isinstance(v, float) and not float(v).is_integer() else str(int(v)) if isinstance(v, (int, float)) else v), C.CHECKPOINTS.get(k, "")) for k, v in s.items()]
    parts.append(md(pd.DataFrame(rows, columns=["item", "this run", "expected"])) + "\n")
keep = ["label", "scheme", "auc", "ndcg_at_k", "sensitivity_at_k", "precision_at_k",
        "hits_total", "top_total", "ppv_pooled", "hits_min_seed", "hits_max_seed"]
for name, title in [("02_replicate_summary.csv", "2. Main replication"),
                    ("03_serial_summary.csv", "3. Serial-fraud schemes (RUSBoost-raw28)"),
                    ("04_extension_summary.csv", "4. Extension: GBM and ratio features")]:
    f = out(name)
    if f.exists():
        d = pd.read_csv(f)
        parts.append(f"## {title}\n")
        parts.append(md(d[[c for c in keep if c in d.columns]]) + "\n")
for name, title in [("02_replicate_by_year.csv", "Appendix A. Main replication by year")]:
    f = out(name)
    if f.exists():
        parts.append(f"## {title}\n")
        parts.append(md(pd.read_csv(f)) + "\n")
f = out("05_tuning.csv")
if f.exists():
    t = pd.read_csv(f).groupby("n_trees")["auc"].mean().reset_index()
    parts.append("## 5. Tuning (train 1991-1999, validate 2001)\n")
    parts.append(md(t) + "\n")
imps = sorted(C.RESULTS_DIR.glob("06_importance_*.csv"))
for f in imps:
    parts.append(f"## 6. Permutation importance: {f.stem}\n")
    parts.append(md(pd.read_csv(f).head(10)) + "\n")
parts.append("Checkpoints (Walker 2021, Table 2): published 16 hits / PPV 4.5%; re-run 15; "
             "recoded with p_aaer 9 hits / PPV 2.5%; logit benchmark PPV 2.63%.\n")
(out("REPORT.md")).write_text("\n".join(parts), encoding="utf-8")
print("Wrote", out("REPORT.md"))
