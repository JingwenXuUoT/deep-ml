import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    x = np.asarray(logits, dtype=float)

    # stable softmax
    e = np.exp(x-np.max(x))
    probs = e / e.sum()
    
    # descending by prob, ties broken by smaller index(stable sort on negated probs)
    order = np.argsort(-probs, kind='stable')
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)

    # 1st position where cumulative mass reaches p; nucleus is everything up to it
    # tolerance guards against float noise when cum lands exactly on p (e.g. p=1.0)
    k = int(np.searchsorted(cum, p-1e-12, side='left')) + 1
    k = min(k, len(probs))

    keep = order[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    out /= out.sum()

    return [round(float(v), 4) for v in out]