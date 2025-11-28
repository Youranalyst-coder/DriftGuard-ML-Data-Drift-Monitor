import pandas as pd
from drift_metrics import population_stability_index, ks_statistic, kl_divergence

NUMERIC_DTYPES = ["int64", "float64"]

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                 psi_threshold: float = 0.2,
                 ks_threshold: float = 0.1):
    """
    Returns a dataframe with drift metrics for each numeric feature.
    """
    common_cols = [c for c in reference_df.columns if c in current_df.columns]

    results = []

    for col in common_cols:
        if str(reference_df[col].dtype) not in NUMERIC_DTYPES:
            # For simplicity, only numeric here (you can extend to categoricals later)
            continue

        ref = reference_df[col]
        cur = current_df[col]

        psi = population_stability_index(ref, cur)
        ks_stat, ks_pval = ks_statistic(ref, cur)
        kl = kl_divergence(ref, cur)

        drift_flag = False
        reasons = []

        if psi is not None and psi >= psi_threshold:
            drift_flag = True
            reasons.append(f"PSI={psi:.2f} ≥ {psi_threshold}")

        if ks_stat is not None and ks_stat >= ks_threshold:
            drift_flag = True
            reasons.append(f"KS={ks_stat:.2f} ≥ {ks_threshold}")

        results.append({
            "feature": col,
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "kl_divergence": kl,
            "drift": drift_flag,
            "reasons": "; ".join(reasons)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="psi", ascending=False, na_position="last")

    overall_drift = result_df["drift"].any() if not result_df.empty else False
    return overall_drift, result_df
