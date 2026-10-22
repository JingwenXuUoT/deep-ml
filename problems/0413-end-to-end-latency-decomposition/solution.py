import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    stages = list(stage_latencies.keys())
    arrays = {s: np.asarray(stage_latencies[s], dtype=float) for s in stages}
    e2e = np.sum([arrays[s] for s in stages], axis = 0)

    e2e_mean = float(np.mean(e2e))
    e2e_percentiles = {p: round(float(np.percentile(e2e, p)), 2) for p in percentiles}

    stage_stats = {}
    stage_means = {}
    for s in stages:
        a = arrays[s]
        m = float(np.mean(a))
        stage_means[s] = m
        stage_stats[s] = {
            'mean': round(m, 2),
            'std': round(float(np.std(a)), 2),
            'percentiles': {
                p: round(float(np.percentile(a, p)), 2) for p in percentiles
            },
        }
    bottleneck = max(stages, key=lambda s: stage_means[s])

    
    # Use unrounded means for the percentages to avoid compounding rounding error
    stage_pct = {
        s: round(stage_means[s] / e2e_mean * 100, 2) if e2e_mean != 0 else 0.0 for s in stages
    }

    return {
        'e2e_mean': round(e2e_mean, 2),
        'e2e_percentiles': e2e_percentiles,
        'stage_stats': stage_stats,
        'bottleneck': bottleneck,
        'stage_pct': stage_pct,
    }
