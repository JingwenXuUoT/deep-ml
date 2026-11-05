import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    # Convert inputs to numpy arrays for efficient computation
    log_probs_arr = np.array(log_probs)
    correct_arr = np.array(correct_answers)
    
    # 1. Determine predicted answer for each question (choice with highest log-probability)
    predictions = np.argmax(log_probs_arr, axis=1)
    
    # 2. Calculate overall accuracy
    accuracy = float(np.mean(predictions == correct_arr))
    
    # 3. Compute normalized probabilities with numerical stability (Softmax trick)
    max_log_probs = np.max(log_probs_arr, axis=1, keepdims=True)
    exp_log_probs = np.exp(log_probs_arr - max_log_probs)
    probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
    
    # Extract the probability assigned to the correct answer for each question
    correct_probs = probs[np.arange(len(correct_arr)), correct_arr]
    avg_correct_prob = float(np.mean(correct_probs))
    
    return {
        'accuracy': accuracy,
        'predictions': predictions.tolist(),
        'avg_correct_prob': round(avg_correct_prob, 4)
    }