def compute_inference_metrics(timestamps: list[float]) -> dict:
	"""
	Compute LLM inference performance metrics from token timestamps.
	
	Args:
		timestamps: List of floats where timestamps[0] is the request start time
		            and timestamps[1:] are the times when each output token was generated.
	
	Returns:
		Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
	"""
	res = {'ttft': 0.0, 'tps': 0.0, 'itl': 0.0}
    n_tokens = len(timestamps) - 1
    if n_tokens < 1:
        return res

    start = timestamps[0]
    total = timestamps[-1] - start

    res['ttft'] = timestamps[1] - start
    res['tps'] = n_tokens / total if total > 0 else 0.0
    if n_tokens > 1:
        res['itl'] = (timestamps[-1] - timestamps[1]) / (n_tokens - 1)

    return res