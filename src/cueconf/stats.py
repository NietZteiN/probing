"""Statistics (plan §4.8): bootstrap CIs over matched sets, clustered logistic regression,
FDR across layer x position tests, crossover step, instance-level margin -> error link.

Mixed-effects logistic regression with a random intercept per problem template is the plan's
wording; statsmodels' mixed GLM is Bayesian-approximate and slow, so the PRIMARY analysis is a
logistic regression with cluster-robust standard errors clustered on the matched set (which is
what the random intercept is for), and `mixed_logit` is provided for the appendix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.mean) -> tuple[float, float, float]:
    """Cluster bootstrap: resample matched sets with replacement. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float); clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [values[inv == i] for i in range(len(uniq))]
    point = float(stat(values))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by), len(by))
        boots.append(stat(np.concatenate([by[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_difference_ci(df: pd.DataFrame, cond_a: str, cond_b: str, col: str = "correct", n_boot: int = 2000, seed: int = 0):
    """Within-set difference a - b (e.g. interference = neutral - incongruent)."""
    a = df[df.condition == cond_a].set_index("set_id")[col].astype(float)
    b = df[df.condition == cond_b].set_index("set_id")[col].astype(float)
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).values
    return bootstrap_ci(d, np.asarray(common), n_boot, seed)


def clustered_logit(df: pd.DataFrame, formula: str, cluster: str = "set_id"):
    import statsmodels.formula.api as smf
    model = smf.logit(formula, data=df)
    res = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df[cluster].astype("category").cat.codes})
    return res


def mixed_logit(df: pd.DataFrame, formula: str, group: str = "set_id"):
    """Appendix: random intercept per matched set (statsmodels BinomialBayesMixedGLM)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    vc = {"set": f"0 + C({group})"}
    m = BinomialBayesMixedGLM.from_formula(formula, vc, df)
    return m.fit_vb()


def fdr(pvals: np.ndarray, alpha: float = 0.05):
    from statsmodels.stats.multitest import multipletests
    rej, p_adj, _, _ = multipletests(np.asarray(pvals), alpha=alpha, method="fdr_bh")
    return rej, p_adj


def crossover_step(margin_by_step: dict[str, float], order: list[str]) -> str | None:
    """First position (in `order`) at which the mean margin is positive AND stays positive."""
    vals = [margin_by_step.get(p) for p in order]
    for i, p in enumerate(order):
        tail = [v for v in vals[i:] if v is not None]
        if tail and all(v > 0 for v in tail):
            return p
    return None


def t_star(acc_by_pos: dict[int, float], tau: float = 0.9) -> int | None:
    """Kudo et al. eq. (2): first position where max-over-layers accuracy exceeds tau."""
    for t in sorted(acc_by_pos):
        if acc_by_pos[t] > tau:
            return t
    return None
