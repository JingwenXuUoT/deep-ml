import numpy as np

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark.
    
    Args:
        test_cases: List of dicts with keys:
            - 'expected': Expected output string
            - 'actual': Actual output string (or None if execution failed)
            - 'status': 'success', 'error', or 'timeout'
        numeric_tolerance: Tolerance for floating-point comparisons
        
    Returns:
        Dict with keys:
            - 'pass_rate': Proportion of passed tests (float, rounded to 4 decimals)
            - 'error_rate': Proportion of execution errors (float, rounded to 4 decimals)
            - 'passed_count': Number of passed tests (int)
            - 'total_count': Total number of tests (int)
            - 'verdicts': List of 'pass', 'fail', or 'error' for each test
    """
    total_count = len(test_cases)
    
    # Handle empty test case lists appropriately
    if total_count == 0:
        return {
            'pass_rate': 0.0,
            'error_rate': 0.0,
            'passed_count': 0,
            'total_count': 0,
            'verdicts': []
        }
    passed_count = 0
    error_count = 0
    verdicts = []
    
    for tc in test_cases:
        status = tc.get('status')
        # 1. Mark tests with non-success status as 'error'
        if status != 'success':
            verdicts.append('error')
            error_count += 1
            continue
        
        expected_str = tc.get('expected', '')
        actual_str = tc.get('actual')

        # Guard against None actual value for successful execution (as safety precaution)
        if actual_str is None:
            verdicts.append('error')
            error_count += 1
            continue
        
        # 2. For successful executions, compare outputs after stripping whitespace
        expected_stripped = expected_str.strip()
        actual_stripped = actual_str.strip()

        # 3. Support numeric tolerance for floating-point comparisons (try parsing as floats first)
        try:
            expected_num = float(expected_stripped)
            actual_num = float(actual_stripped)

            # Use absolute distance check within numerical tolerance limits
            if abs(expected_num - actual_num) <= numeric_tolerance:
                verdicts.append('pass')
                passed_count += 1
            else:
                verdicts.append('fail')
            
        except ValueError:
            # 4. Fall back to exact string matching for non-numeric outputs
            if expected_stripped == actual_stripped:
                verdicts.append('pass')
                passed_count += 1
            else:
                verdicts.append('fail')   

    pass_rate = passed_count / total_count
    error_rate = error_count / total_count

    return {
        'pass_rate': round(float(pass_rate), 4),
        'error_rate': round(float(error_rate), 4),
        'passed_count': int(passed_count),
        'total_count': int(total_count),
        'verdicts': verdicts
    }