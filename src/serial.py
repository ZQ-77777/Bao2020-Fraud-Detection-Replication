"""Serial fraud spanning training and test periods (Bao et al. 2020, Section 3.3)."""
from .data import clean_ids

SCHEMES = {
    # name: (id column, action)
    "none": (None, None),                            # no adjustment: leakage benchmark
    "recode_new_p_aaer": ("new_p_aaer", "recode"),   # authors' original code (pre-erratum)
    "recode_p_aaer": ("p_aaer", "recode"),           # erratum / Walker (2021) correction
    "drop_p_aaer": ("p_aaer", "drop"),               # drop rows instead of relabeling them as 0
    "drop_firm": ("gvkey", "drop"),                  # strictest: drop training fraud years of any test-year fraud firm
}


def apply(train, test, scheme):
    """Return (adjusted training data, number of training rows affected)."""
    id_col, action = SCHEMES[scheme]
    if id_col is None:
        return train, 0
    if id_col not in train.columns:
        raise KeyError(f"Scheme '{scheme}' needs column '{id_col}', which this data file does not have.")
    if id_col == "gvkey":
        test_ids = set(test.loc[test["misstate"] == 1, id_col].dropna())
        train_ids = train[id_col]
    else:
        test_ids = set(clean_ids(test.loc[test["misstate"] == 1, id_col]).dropna())
        train_ids = clean_ids(train[id_col])
    mask = train_ids.isin(test_ids) & (train["misstate"] == 1)
    n = int(mask.sum())
    if action == "recode":
        train = train.copy()
        train.loc[mask, "misstate"] = 0
    else:
        train = train.loc[~mask]
    return train, n


def available(df):
    return [name for name, (col, _) in SCHEMES.items() if col is None or col in df.columns]
