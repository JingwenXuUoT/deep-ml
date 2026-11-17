def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    if not comparisons:
        return {
            'results': [],
            'win_rate_a': 0.0,
            'win_rate_b': 0.0,
            'tie_rate': 0.0,
            'avg_margin': 0.0,
        }
    total_w = sum(criteria_weights.values())
    norm = {c: w / total_w for c, w in criteria_weights.items()} if total_w else {
        c: 0.0 for c in criteria_weights
    }
    results = []
    n_a = n_b = n_tie = 0
    margin_sum = 0.0

    for comp in comparisons:
        sa, sb = comp['scores_a'], comp['scores_b']
        wa = sum(norm[c] * sa.get(c, 0) for c in norm)
        wb = sum(norm[c] * sb.get(c, 0) for c in norm)

        diff = wa - wb
        margin = abs(diff)

        if round(margin, 4) <= tie_threshold:
            winner = 'tie'
            n_tie += 1
        elif diff > 0:
            winner = 'A'
            n_a += 1
        else:
            winner = 'B'
            n_b += 1
        
        margin_sum += margin
        results.append({'id': comp['id'], 'winner': winner, 'margin': round(margin, 4)})
    n = len(comparisons)
    return {
        'results': results,
        'win_rate_a': round(n_a / n, 4),
        'win_rate_b': round(n_b / n, 4),
        'tie_rate': round(n_tie / n, 4),
        'avg_margin': round(margin_sum / n, 4),
    }