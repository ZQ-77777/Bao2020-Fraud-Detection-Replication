import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default=str(C.DEFAULT_DATA), help="path to the CSV from JarFraud/FraudDetection")
    p.add_argument("--trees", type=int, default=None, help="RUSBoost trees (default 3000; use e.g. 300 for a quick run)")
    p.add_argument("--seeds", type=int, default=1, help="number of random seeds to average over")
    p.add_argument("--years", type=str, default=None, help="test years, e.g. 2003-2008 or 2003,2005")
    p.add_argument("--common-sample", action="store_true",
                   help="keep only rows with all 28 raw items and 14 ratios, so every model uses the same sample")
    p.add_argument("--leaves", type=int, default=None, help="RUSBoost max leaf nodes per tree (default 11; 0 = unlimited)")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes (-1 = all CPU cores)")
    return p


def parse_years(s):
    if not s:
        return C.TEST_YEARS
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def out(name):
    return C.RESULTS_DIR / name
