"""Project-wide settings for the Bao et al. (2020, JAR) replication."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Default data file (from github.com/JarFraud/FraudDetection).
# The current repo version is data_FraudDetection_JAR2020.csv (post-erratum, uses p_aaer).
# Older versions shipped uscecchini28.csv, which also contains new_p_aaer.
DEFAULT_DATA = DATA_DIR / "data_FraudDetection_JAR2020.csv"

# ---- Sample design (Bao et al. 2020, Section 3; run_RUSBoost.m) ----
TRAIN_START = 1991          # first training year
TEST_YEARS = list(range(2003, 2009))   # out-of-sample test years 2003-2008
GAP = 2                     # train on 1991..(test_year - GAP); AAERs surface with a lag
TOP_K = 0.01                # evaluate the top 1% of firms ranked by fraud score

# ---- RUSBoost hyper-parameters (Matlab: fitensemble(..., 'RUSBoost', 3000, tree, 'LearnRate', 0.1)) ----
RUS_N_ESTIMATORS = 3000
RUS_LEARNING_RATE = 0.1
RUS_MIN_LEAF = 5
RUS_VARIANT = "m2"          # AdaBoost.M2-style pseudo-loss (closest to Matlab); "m1" = discrete AdaBoost
RUS_MAX_LEAF_NODES = 11     # Matlab boosting default: MaxNumSplits = 10  -> 11 leaves (assumption, see README)

# Tuning design from the authors' README: train 1991-1999, validate on 2001
TUNE_TRAIN_END = 1999
TUNE_VALID_YEAR = 2001
TUNE_GRID = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]

# ---- Checkpoints to compare against (Walker 2021, Econ Journal Watch 18(1), Table 2) ----
CHECKPOINTS = {
    "n_obs": 146045,
    "n_fraud_firm_years": 964,
    "n_unique_aaer": 413,
    "hits_top1pct_published": 16,      # 2003-2008 total, published model
    "ppv_top1pct_published": 0.045,
    "hits_top1pct_rerun": 15,          # Walker re-running authors' code/data
    "hits_top1pct_recoded": 9,         # Walker using original p_aaer ids
    "ppv_top1pct_recoded": 0.025,
    "ppv_top1pct_logit_benchmark": 0.0263,
}
