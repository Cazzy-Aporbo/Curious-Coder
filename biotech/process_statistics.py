"""Descriptive process-drift diagnostics with explicit inferential eligibility."""

import numpy as np
from scipy.stats import chi2_contingency


def window_diagnostics(reference, window, edges, independent_samples=False):
    reference, window, edges = (np.asarray(value, dtype=float) for value in (reference, window, edges))
    if any(values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() for values in (reference, window)):
        raise ValueError("Reference and current windows must contain at least two finite observations.")
    if edges.ndim != 1 or len(edges) < 3 or not np.all(np.diff(edges) > 0) or edges[0] != -np.inf or edges[-1] != np.inf:
        raise ValueError("Use fixed ordered bins with explicit underflow/overflow coverage.")
    reference_counts = np.histogram(reference, edges)[0]
    current_counts = np.histogram(window, edges)[0]
    probabilities = current_counts / current_counts.sum()
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log2(probabilities[positive])))
    table = np.vstack([reference_counts, current_counts])
    retained = table.sum(axis=0) > 0
    reasons = []
    statistic, p_value, minimum_expected = None, None, None
    if retained.sum() < 2:
        reasons.append("LESS_THAN_TWO_OCCUPIED_BINS")
    else:
        statistic, computed_p, _, expected = chi2_contingency(table[:, retained], correction=False)
        minimum_expected = float(expected.min())
        if minimum_expected < 5:
            reasons.append("EXPECTED_CELL_COUNT_BELOW_FIVE")
        if not independent_samples:
            reasons.append("INDEPENDENCE_NOT_ESTABLISHED")
        if not reasons:
            p_value = float(computed_p)
        statistic = float(statistic)
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = 1.4826 * mad
    shift = float(np.median(window) - median)
    robust_shift = shift / scale if scale > 0 else None
    if scale == 0:
        reasons.append("ZERO_REFERENCE_MAD")
    return {"reference_n": len(reference), "window_n": len(window), "reference_counts": reference_counts.tolist(),
            "window_counts": current_counts.tolist(), "shannon_entropy_bits": entropy,
            "sample_variance": float(window.var(ddof=1)), "median_shift": shift, "reference_mad_scale": scale,
            "robust_median_shift": robust_shift, "chi_square_homogeneity_statistic": statistic,
            "chi_square_p_value": p_value, "minimum_expected_cell_count": minimum_expected,
            "inferential_cautions": reasons,
            "drift_review": robust_shift is not None and abs(robust_shift) > 4,
            "scope": "Distribution/process review only; not a contamination diagnosis, future-yield prediction, or validated alarm policy."}


def rolling_diagnostics(values, reference_count=48, window_size=16):
    values = np.asarray(values, dtype=float)
    if reference_count < 8 or window_size < 2 or len(values) < reference_count + window_size:
        raise ValueError("Insufficient observations for the separate baseline and current windows.")
    reference = values[:reference_count]
    edges = np.concatenate(([-np.inf], np.quantile(reference, [.25, .5, .75]), [np.inf]))
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Baseline quantiles collapse; inspect sensor resolution or specify a different qualified binning policy.")
    rows = []
    for end in range(reference_count + window_size, len(values) + 1):
        result = window_diagnostics(reference, values[end - window_size:end], edges, independent_samples=False)
        rows.append({"ending_observation": end - 1, **result})
    return {"reference_count": reference_count, "window_size": window_size,
            "bin_edges": [str(value) for value in edges], "windows": rows,
            "protocol": "Baseline-fitted quartile bins are frozen; overlapping serial windows do not establish independent multinomial sampling."}
