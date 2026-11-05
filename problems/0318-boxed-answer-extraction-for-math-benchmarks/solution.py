def extract_boxed_answer(response: str) -> str:
    """
    Extract the answer from within \boxed{...} in a model response.
    
    Args:
        response: The model's text response containing a boxed answer
    
    Returns:
        The content inside the last \boxed{}, or empty string if not found
    """
    target = r"\boxed{"
    last_boxed_idx = response.rfind(target)

    if last_boxed_idx == -1:
        return ""
    
    # Start prasing right after the opening brace of the last \boxed{
    start_idx = last_boxed_idx + len(target)
    brace_count = 1
    content_chars = []

    for i in range(start_idx, len(response)):
        char = response[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        # Once the matching closing brace is found, return the content
        if brace_count == 0:
            return "".join(content_chars)
        
        content_chars.append(char)
    
    # If braces are unbalanced ans never closed properly, return an empty string
    return ""