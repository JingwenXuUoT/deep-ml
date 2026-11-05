import numpy as np

def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0
) -> dict:
    """
    Evaluate LLM response using rubric-based multi-judge scoring.
    
    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion
    
    Returns:
        Dictionary with evaluation results
    """
    scores_arr = np.array(judge_scores) # Shape: (J, C) where J = judges, C = criteria
    weights_arr = np.array(criteria_weights) # Shape: (C,)

    # 1. Compute means and standard deviations across judges for each criterion (axis=0)
    # Spec states: population standard deviation with ddof=0
    criterion_means = np.mean(scores_arr, axis=0)
    criterion_stds = np.std(scores_arr, axis=0, ddof=0)

    # 2. Calculate weighted score: S = sum(w_j * mu_j)
    weighted_score = np.sum(weights_arr * criterion_means)

    # 3. Calculate normalized score
    normalized_score = weighted_score / max_score

    # 4. Determine pass status (True if normalized score >= passing_threshold)
    # The condition uses the unrounded score as per the prompt instructions
    pass_status = bool(normalized_score >= passing_threshold)

    # 5. Compute judge agreement metrics
    J = scores_arr.shape[0]
    if J <= 1:
        judge_agreement = 1.0
    else:
        mean_sigma = np.mean(criterion_stds)
        judge_agreement = max(0.0, 1.0 - (mean_sigma / (max_score / 2.0)))

    return {
        'criterion_scores': [round(float(mu), 4) for mu in criterion_means],
        'weighted_score': round(float(weighted_score), 4),
        'normalized_score': round(float(normalized_score), 4),
        'pass_status': pass_status,
        'judge_agreement': round(float(judge_agreement), 4)
    }