"""Metamer analysis: Bayesian and permutation-based inference.

Replaces the binary "reconstruction similarity ≥ baseline → metamer" rule with
three complementary, continuous measures:

    1. bayes_factor_directional  – parametric BF+0 on paired differences
    2. bayes_factor_ranks        – nonparametric BF+0 on jointly ranked scores
    3. permutation_test          – exact Monte Carlo p-values (paired & rank)

Hypothesis framing
------------------
Given n_views paired scores — one per camera view — at the end of optimisation:

    d_i = reconstruction_sim_i - baseline_sim_i,  i = 0 … n_views-1

H0: μ_d < 0   (reconstruction is on average worse than baseline)
H+: μ_d ≥ 0   (reconstruction is at least as good as baseline)

References
----------
Rouder et al. (2009)       — JZS Bayes factor for the t-test.
Morey & Wagenmakers (2014) — directional BFs from two-sided ones.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import rankdata, ttest_1samp, ttest_ind
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# Jeffreys evidence scale
# ---------------------------------------------------------------------------

_SCALE = [
    (100,    "Decisive (H+)"),
    (30,     "Very strong (H+)"),
    (10,     "Strong (H+)"),
    (3,      "Moderate (H+)"),
    (1,      "Anecdotal (H+)"),
    (1 / 3,  "Anecdotal (H0)"),
    (1 / 10, "Moderate (H0)"),
    (1 / 30, "Strong (H0)"),
]


def jeffreys_label(bf: float) -> str:
    """Map a Bayes factor to its Jeffreys (1961) evidence-scale label."""
    for threshold, label in _SCALE:
        if bf >= threshold:
            return label
    return "Decisive (H0)"


# ---------------------------------------------------------------------------
# JZS integration helpers
# ---------------------------------------------------------------------------

def _jzs_bf10_one_sample(t_stat: float, n: int, r: float) -> float:
    """Two-sided JZS BF10 for a one-sample t-test (Rouder et al., 2009).

    Prior: δ ~ Cauchy(0, r) via Inv-Gamma(1/2, 1/2) scale mixture.
    Integrand is computed in log-space for numerical stability.
    """
    df = float(n - 1)
    t2 = float(t_stat) ** 2

    def integrand(g: float) -> float:
        denom = 1.0 + n * g * r ** 2
        log_lr = (
            -0.5 * np.log(denom)
            - (df + 1) / 2 * np.log1p(t2 / (df * denom))
        )
        log_prior_g = -0.5 * np.log(2 * np.pi) - 1.5 * np.log(g) - 1.0 / (2.0 * g)
        return np.exp(log_lr + log_prior_g)

    numerator, _ = quad(integrand, 0, np.inf, limit=200, epsabs=1e-10, epsrel=1e-8)
    log_h0 = -(df + 1) / 2 * np.log1p(t2 / df)
    return numerator / np.exp(log_h0)


def _jzs_bf10_two_sample(t_stat: float, n1: int, n2: int, r: float) -> float:
    """Two-sided JZS BF10 for an independent two-sample t-test (Rouder et al., 2009).

    Uses the harmonic-mean effective n: n_eff = n1·n2 / (n1+n2).
    """
    df = float(n1 + n2 - 2)
    n_eff = n1 * n2 / (n1 + n2)
    t2 = float(t_stat) ** 2

    def integrand(g: float) -> float:
        denom = 1.0 + n_eff * g * r ** 2
        log_lr = (
            -0.5 * np.log(denom)
            - (df + 1) / 2 * np.log1p(t2 / (df * denom))
        )
        log_prior_g = -0.5 * np.log(2 * np.pi) - 1.5 * np.log(g) - 1.0 / (2.0 * g)
        return np.exp(log_lr + log_prior_g)

    numerator, _ = quad(integrand, 0, np.inf, limit=200, epsabs=1e-10, epsrel=1e-8)
    log_h0 = -(df + 1) / 2 * np.log1p(t2 / df)
    return numerator / np.exp(log_h0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _validate_pair(recon: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recon = np.asarray(recon, dtype=float).ravel()
    base  = np.asarray(base,  dtype=float).ravel()
    if len(recon) != len(base):
        raise ValueError(
            f"reconstruction and baseline must have the same length "
            f"(got {len(recon)} vs {len(base)})"
        )
    return recon, base


def bayes_factor_directional(
    reconstruction: np.ndarray,
    baseline: np.ndarray,
    r: float = float(np.sqrt(2) / 2),
) -> dict:
    """Parametric directional BF+0 on paired per-view differences.

    H0: μ(reconstruction − baseline) < 0
    H+: μ(reconstruction − baseline) ≥ 0

    Parameters
    ----------
    reconstruction : array-like, shape (n_views,)
    baseline       : array-like, shape (n_views,)
    r              : Cauchy prior scale (default √2/2 ≈ 0.707).

    Returns
    -------
    dict
        bf_plus0  – BF+0; > 1 supports H+.
        log10_bf  – log₁₀(BF+0).
        label     – Jeffreys scale category.
        t_stat    – one-sample t on d = reconstruction − baseline.
        p_value   – two-sided p (informational).
        mean_diff – mean(d).
        std_diff  – std(d, ddof=1).
        n         – n_views.
    """
    recon, base = _validate_pair(reconstruction, baseline)
    diff = recon - base
    n    = len(diff)

    t_stat = float(ttest_1samp(diff, popmean=0.0).statistic)
    bf10   = _jzs_bf10_one_sample(t_stat, n, r)

    # Directional correction (Morey & Wagenmakers 2014):
    # BF+0 = 2 × BF10 × P(δ > 0 | data, H₁)
    p_positive = float(t_dist.cdf(t_stat, df=n - 1))
    bf_plus0   = 2.0 * bf10 * p_positive
    log10_bf   = float(np.log10(bf_plus0)) if bf_plus0 > 0 else float("-inf")

    return {
        "bf_plus0":  bf_plus0,
        "log10_bf":  log10_bf,
        "label":     jeffreys_label(bf_plus0),
        "t_stat":    t_stat,
        "p_value":   float(ttest_1samp(diff, popmean=0.0).pvalue),
        "mean_diff": float(diff.mean()),
        "std_diff":  float(diff.std(ddof=1)),
        "n":         n,
    }


def bayes_factor_ranks(
    reconstruction: np.ndarray,
    baseline: np.ndarray,
    r: float = float(np.sqrt(2) / 2),
) -> dict:
    """Nonparametric directional BF+0 on jointly ranked scores.

    All 2·n_views scores are pooled and ranked together without knowledge of
    group membership (ties → average rank).  Tests whether reconstruction
    scores hold higher ranks on average.

    H0: mean_rank(reconstruction) ≤ mean_rank(baseline)
    H+: mean_rank(reconstruction) > mean_rank(baseline)

    Parameters
    ----------
    reconstruction : array-like, shape (n_views,)
    baseline       : array-like, shape (n_views,)
    r              : Cauchy prior scale (default √2/2).

    Returns
    -------
    dict
        bf_plus0         – BF+0 on ranks; > 1 supports H+.
        log10_bf         – log₁₀(BF+0).
        label            – Jeffreys scale category.
        t_stat           – two-sample t on (recon_ranks, base_ranks).
        p_value          – two-sided p (informational).
        mean_rank_recon  – mean rank of reconstruction scores.
        mean_rank_base   – mean rank of baseline scores.
        n                – n_views.
    """
    recon, base = _validate_pair(reconstruction, baseline)
    n = len(recon)

    all_ranks   = rankdata(np.concatenate([recon, base]))
    recon_ranks = all_ranks[:n]
    base_ranks  = all_ranks[n:]

    t_stat = float(ttest_ind(recon_ranks, base_ranks).statistic)
    bf10   = _jzs_bf10_two_sample(t_stat, n, n, r)

    p_positive = float(t_dist.cdf(t_stat, df=2 * n - 2))
    bf_plus0   = 2.0 * bf10 * p_positive
    log10_bf   = float(np.log10(bf_plus0)) if bf_plus0 > 0 else float("-inf")

    return {
        "bf_plus0":        bf_plus0,
        "log10_bf":        log10_bf,
        "label":           jeffreys_label(bf_plus0),
        "t_stat":          t_stat,
        "p_value":         float(ttest_ind(recon_ranks, base_ranks).pvalue),
        "mean_rank_recon": float(recon_ranks.mean()),
        "mean_rank_base":  float(base_ranks.mean()),
        "n":               n,
    }


def permutation_test(
    reconstruction: np.ndarray,
    baseline: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict:
    """Exact Monte Carlo permutation test for H+: reconstruction ≥ baseline.

    Two null distributions are built simultaneously:

    Paired (sign-flip)
        Under H0 of within-pair exchangeability, each difference d_i =
        reconstruction_i − baseline_i is equally likely to be positive or
        negative.  Each permutation randomly flips a subset of signs.
        Test statistic: mean(d).

    Rank (group-reassignment)
        Under H0, all 2·n pooled ranks are exchangeable between groups.
        Each permutation randomly assigns n ranks to "reconstruction" and n
        to "baseline".
        Test statistic: mean_rank(reconstruction) − mean_rank(baseline).

    Both are vectorised: no Python loop over permutations.

    Parameters
    ----------
    reconstruction : array-like, shape (n_views,)
    baseline       : array-like, shape (n_views,)
    n_permutations : number of Monte Carlo draws (default 10 000).
    seed           : RNG seed for reproducibility.

    Returns
    -------
    dict
        p_paired        – one-sided p-value from paired permutation.
        p_rank          – one-sided p-value from rank permutation.
        obs_mean_diff   – observed mean(d).
        obs_rank_diff   – observed mean_rank(recon) − mean_rank(base).
        null_mean_diff  – (n_permutations,) null distribution of mean(d).
        null_rank_diff  – (n_permutations,) null distribution of rank diff.
        n_permutations  – as passed.
        n               – n_views.
    """
    recon, base = _validate_pair(reconstruction, baseline)
    rng = np.random.default_rng(seed)
    n   = len(recon)

    diff      = recon - base
    obs_mean  = float(diff.mean())

    all_ranks    = rankdata(np.concatenate([recon, base]))
    obs_rank_diff = float(all_ranks[:n].mean() - all_ranks[n:].mean())

    # Paired permutation: flip random subsets of signs — shape (n_perm, n)
    signs          = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    null_mean_diff = (diff * signs).mean(axis=1)

    # Rank permutation: randomly split 2n ranks into two groups of n
    shuffle_idx    = rng.random((n_permutations, 2 * n)).argsort(axis=1)
    null_rank_diff = (
        all_ranks[shuffle_idx[:, :n]].mean(axis=1)
        - all_ranks[shuffle_idx[:, n:]].mean(axis=1)
    )

    p_paired = float((null_mean_diff >= obs_mean).mean())
    p_rank   = float((null_rank_diff >= obs_rank_diff).mean())

    return {
        "p_paired":       p_paired,
        "p_rank":         p_rank,
        "obs_mean_diff":  obs_mean,
        "obs_rank_diff":  obs_rank_diff,
        "null_mean_diff": null_mean_diff,
        "null_rank_diff": null_rank_diff,
        "n_permutations": n_permutations,
        "n":              n,
    }
