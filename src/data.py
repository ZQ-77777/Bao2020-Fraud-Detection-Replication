"""Load and check the Bao et al. (2020) dataset."""
import numpy as np
import pandas as pd

RAW28 = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
         "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
         "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]

RATIO14 = ["dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs", "ch_cm",
           "ch_roa", "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf"]

# Spelling variants seen across versions / documentation
_ALIASES = {"gkvey": "gvkey", "soft_assset": "soft_assets", "soft_asset": "soft_assets",
            "ebit": "EBIT"}


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _ALIASES.items()
                            if k in df.columns and v not in df.columns})
    for col in ["fyear", "misstate"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found. First columns: {list(df.columns)[:15]}")
    missing_raw = [c for c in RAW28 if c not in df.columns]
    if missing_raw:
        raise KeyError(f"Missing raw accounting variables: {missing_raw}")
    df["fyear"] = df["fyear"].astype(int)
    df["misstate"] = (df["misstate"] != 0).astype(int)
    ratios = [c for c in RATIO14 if c in df.columns]
    df[ratios] = df[ratios].replace([np.inf, -np.inf], np.nan)
    return df


def id_columns(df):
    """AAER identifier columns available in this file version."""
    return [c for c in ["p_aaer", "new_p_aaer"] if c in df.columns]


def ratio_columns(df):
    return [c for c in RATIO14 if c in df.columns]


def clean_ids(s):
    """Treat 0 / blank / NaN AAER ids as 'no AAER'."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.fillna(0) != 0)


def split(df, test_year, train_start, gap):
    train = df[(df["fyear"] >= train_start) & (df["fyear"] <= test_year - gap)]
    test = df[df["fyear"] == test_year]
    return train, test


def summarize(df):
    out = {
        "n_obs": len(df),
        "year_min": int(df["fyear"].min()),
        "year_max": int(df["fyear"].max()),
        "n_fraud_firm_years": int(df["misstate"].sum()),
        "fraud_rate": float(df["misstate"].mean()),
    }
    if "p_aaer" in df.columns:
        fr = df[df["misstate"] == 1]
        ids = clean_ids(fr["p_aaer"])
        out["n_unique_aaer"] = int(ids.nunique())
        dur = fr.assign(pid=ids).groupby("pid")["fyear"].nunique()
        out["fraud_duration_mean"] = float(dur.mean())
        out["fraud_duration_median"] = float(dur.median())
        out["fraud_duration_p90"] = float(dur.quantile(0.9))
    out["n_rows_missing_raw28"] = int(df[RAW28].isna().any(axis=1).sum())
    return out


def compare_id_versions(df):
    """AAERs whose new_p_aaer splits one p_aaer into several ids (the issue raised by Walker 2021)."""
    if not {"p_aaer", "new_p_aaer"}.issubset(df.columns):
        return None
    f = df[df["misstate"] == 1].copy()
    f["p"] = clean_ids(f["p_aaer"])
    f["np"] = clean_ids(f["new_p_aaer"])
    g = f.groupby("p").agg(n_new_ids=("np", "nunique"),
                           years=("fyear", lambda x: " ".join(map(str, sorted(set(x))))))
    return g[g["n_new_ids"] > 1].reset_index()
