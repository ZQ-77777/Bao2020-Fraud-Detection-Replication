"""Models: RUSBoost (as in Bao et al. 2020), logit benchmarks, and a gradient-boosting extension."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RUSBoost:
    """Random under-sampling + boosting (Seiffert et al. 2010), as in Matlab's 'RUSBoost'.

    Each round keeps all frauds and draws the same number of non-frauds at random
    (Matlab 'RatioToSmallest' = [1 1]), fits a small tree on that balanced subset with
    the current boosting weights, and updates the weights on the FULL training set.

    variant="m2" (default) follows the AdaBoost.M2 pseudo-loss used by Matlab/Seiffert:
        eps   = sum_i D_i * (1 - p_i),  p_i = tree's probability for the true class
        alpha = lr * log((1 - eps) / eps)
        D_i  <- D_i * exp(-alpha * p_i)          (then renormalize)
        score = sum_t alpha_t * (2 * P_t(fraud | x) - 1)
    variant="m1" uses hard 0/1 errors (discrete AdaBoost).
    Implementation details differ from Matlab, so expect close, not identical, numbers.
    """

    def __init__(self, n_estimators=3000, learning_rate=0.1, min_samples_leaf=5,
                 max_leaf_nodes=11, random_state=0, variant="m2"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.variant = variant

    def _tree_score(self, tree, X):
        if self.variant == "m2":
            proba = tree.predict_proba(X)
            p1 = proba[:, list(tree.classes_).index(1)] if 1 in tree.classes_ else np.zeros(len(X))
            return 2 * p1 - 1          # in [-1, 1]
        return 2 * tree.predict(X) - 1

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        w = np.full(n, 1.0 / n)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        m = min(len(pos), len(neg))
        self.trees_, self.alphas_, self.rounds_ = [], [], []
        self.n_skipped_ = 0
        for t in range(self.n_estimators):
            idx = np.concatenate([rng.choice(pos, m, replace=False) if len(pos) > m else pos,
                                  rng.choice(neg, m, replace=False)])
            tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf,
                                          max_leaf_nodes=self.max_leaf_nodes,
                                          random_state=int(rng.integers(1 << 31)))
            tree.fit(X[idx], y[idx], sample_weight=w[idx] / w[idx].sum())
            h = self._tree_score(tree, X)                 # in [-1, 1]
            p_true = np.where(y == 1, (1 + h) / 2, (1 - h) / 2)
            if self.variant == "m2":
                err = float(np.sum(w * (1 - p_true)) / np.sum(w))
            else:
                err = float(np.sum(w * (p_true < 0.5)) / np.sum(w))
            if err >= 0.5:   # no better than chance on the full weighted sample: skip
                self.n_skipped_ += 1
                continue
            err = max(err, 1e-10)
            alpha = self.learning_rate * np.log((1 - err) / err)
            if self.variant == "m2":
                w = w * np.exp(-alpha * p_true)
            else:
                w = w * np.exp(alpha * (p_true < 0.5))
            w /= w.sum()
            self.trees_.append(tree)
            self.alphas_.append(alpha)
            self.rounds_.append(t + 1)
        return self

    def decision_function(self, X, n_trees=None):
        X = np.asarray(X, dtype=np.float32)
        s = np.zeros(len(X))
        for tree, a, r in zip(self.trees_, self.alphas_, self.rounds_):
            if n_trees is not None and r > n_trees:
                break
            s += a * self._tree_score(tree, X)
        return s

    def staged_scores(self, X, checkpoints):
        """Scores after each number of boosting rounds in `checkpoints` (tuning without refitting)."""
        X = np.asarray(X, dtype=np.float32)
        s = np.zeros(len(X))
        cps = sorted(checkpoints)
        out, j = {}, 0
        for tree, a, r in zip(self.trees_, self.alphas_, self.rounds_):
            while j < len(cps) and r > cps[j]:
                out[cps[j]] = s.copy(); j += 1
            s += a * self._tree_score(tree, X)
        while j < len(cps):
            out[cps[j]] = s.copy(); j += 1
        return out

    def score(self, X):
        return self.decision_function(X)


class Logit:
    """Unpenalized logistic regression (standardized inputs only help convergence)."""

    def __init__(self, random_state=0):
        try:
            lr = LogisticRegression(penalty=None, max_iter=5000)
        except TypeError:   # scikit-learn < 1.2
            lr = LogisticRegression(penalty="none", max_iter=5000)
        self.model = make_pipeline(StandardScaler(), lr)

    def fit(self, X, y):
        self.model.fit(np.asarray(X, dtype=float), y)
        return self

    def score(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]


class GBM:
    """Gradient boosting extension (links to the Geertsema & Lu replication).

    Uses LightGBM if installed, otherwise scikit-learn's HistGradientBoostingClassifier.
    """

    def __init__(self, random_state=0, n_estimators=500, learning_rate=0.03, num_leaves=15):
        self.random_state = random_state
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        spw = (len(y) - y.sum()) / max(y.sum(), 1)
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=self.params["n_estimators"], learning_rate=self.params["learning_rate"],
                num_leaves=self.params["num_leaves"], min_child_samples=20, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw,
                random_state=self.random_state, verbose=-1)
            self.engine = "lightgbm"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=self.params["n_estimators"], learning_rate=self.params["learning_rate"],
                max_leaf_nodes=self.params["num_leaves"], class_weight="balanced",
                random_state=self.random_state)
            self.engine = "sklearn-hgb"
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]


def make(name, seed=0, **kw):
    if name == "rusboost":
        import config as C
        leaves = kw.get("leaves")
        mln = C.RUS_MAX_LEAF_NODES if leaves is None else (None if leaves == 0 else leaves)
        return RUSBoost(n_estimators=kw.get("trees", C.RUS_N_ESTIMATORS), learning_rate=C.RUS_LEARNING_RATE,
                        min_samples_leaf=C.RUS_MIN_LEAF, max_leaf_nodes=mln,
                        random_state=seed, variant=kw.get("variant", C.RUS_VARIANT))
    if name == "logit":
        return Logit(random_state=seed)
    if name == "gbm":
        return GBM(random_state=seed)
    raise ValueError(name)
