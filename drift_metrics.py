import numpy as np
from scipy.stats import ks_2samp, entropy

def population_stability_index(ref, cur, bins=10):
    ref = np.array(ref)
    cur = np.array(cur)

    # Remove NaNs
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    # Same bin edges based on reference
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(ref, quantiles)

    ref_hist, _ = np.histogram(ref, bins=bin_edges)
    cur_hist, _ = np.histogram(cur, bins=bin_edges)

    ref_perc = ref_hist / (len(ref) + 1e-9)
    cur_perc = cur_hist / (len(cur) + 1e-9)

    psi = 0.0
    for r, c in zip(ref_perc, cur_perc):
        if r == 0 or c == 0:
            continue
        psi += (c - r) * np.log(c / r)
    return psi


def ks_statistic(ref, cur):
    ref = np.array(ref)
    cur = np.array(cur)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    stat, pval = ks_2samp(ref, cur)
    return stat, pval


def kl_divergence(ref, cur, bins=20):
    ref = np.array(ref)
    cur = np.array(cur)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    hist_ref, bin_edges = np.histogram(ref, bins=bins, density=True)
    hist_cur, _ = np.histogram(cur, bins=bin_edges, density=True)

    # add small value to avoid log(0)
    hist_ref = hist_ref + 1e-9
    hist_cur = hist_cur + 1e-9

    return entropy(hist_cur, hist_ref)
