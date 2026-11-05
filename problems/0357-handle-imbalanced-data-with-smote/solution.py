import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    n_samples, n_features = X_minority.shape
    k_actual = min(k, n_samples - 1)

    # handle edge case where no neighbors can be picked or no samples are requested
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features))
    
    synthetic_samples = []
    for _ in range(n_synthetic):
        # Draw random values
        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, k_actual)
        gap = np.random.random()

        x_i = X_minority[i]
        # Compute squared Euclidean distances to all samples
        dists = np.sum((X_minority - x_i) ** 2, axis=1)
        # exclude the base sample it self
        dists[i] = np.inf

        # Find the sorted indicies of the closest other minority samples
        nearest_indicies = np.argsort(dists)
        # Select the j-th nearest neighbor
        x_nn = X_minority[nearest_indicies[j]]

        # Synthesize the new sample along the line segment
        new_sample = x_i + gap *(x_nn - x_i)
        synthetic_samples.append(new_sample)
    
    return np.array(synthetic_samples)

