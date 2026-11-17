import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss', 'chosen_rewards', and 'rejected_rewards'
    """
    pc = np.asarray(log_probs_chosen_policy, dtype=float)
    pr = np.asarray(log_probs_rejected_policy, dtype=float)
    rc = np.asarray(log_probs_chosen_ref, dtype=float)
    rr = np.asarray(log_probs_rejected_ref, dtype=float)

    chosen_log_ratio = pc - rc
    rejected_log_ratio = pr - rr

    logits = beta * (chosen_log_ratio - rejected_log_ratio)

    # -log(sigmoid(x)) = softplus(-x) = log(1 + exp(-x))
    loss_per_sample = np.logaddexp(0.0, -logits) # np.logaddexp(0, -x) instead of -np.log(1/(1+np.exp(-x))). The naive form overflows for large negative logits (exp(-x) blows up) and returns inf. logaddexp is stable across the whole range.
    loss = float(np.mean(loss_per_sample))

    im_rewards_chosen = beta * chosen_log_ratio
    im_rewards_rejected = beta * rejected_log_ratio

    return {
        'loss': round(loss, 4),
        'chosen_rewards': [round(float(x), 4) for x in im_rewards_chosen],
        'rejected_rewards': [round(float(x), 4) for x in im_rewards_rejected]
    }