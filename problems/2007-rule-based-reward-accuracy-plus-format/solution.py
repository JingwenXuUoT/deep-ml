# ---- provided: the format reward from the previous problem
import re


def format_reward(response):
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if response.count(tag) != 1:
            return 0.0
    m = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", response, re.DOTALL)
    if not m or not m.group(1).strip() or not m.group(2).strip():
        return 0.0
    return 1.0
# ---- end provided
from fractions import Fraction

def _last_boxed(text):
    marker = "\\boxed{"
    result = None
    start = text.find(marker)
    while start != -1:
        i = start + len(marker)
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            result = text[i:j-1].strip()
        start = text.find(marker, start + 1)
    return result

def extract_answer(response):
    """Answer block text (stripped), or the last \\boxed{...} inside it; None without a complete block."""
    s = response.find("<answer>")
    if s == -1:
        return None
    s += len("<answer>")
    e = response.find("</answer>", s)
    if e == -1:
        return None
    text = response[s:e].strip()
    boxed = _last_boxed(text)
    return boxed if boxed is not None else text


def to_number(text):
    """Fraction for an integer, decimal, a/b or \\frac{a}{b} (optional leading -), else None."""
    t = re.sub(r"\s+", "", text)
    if t.endswith("."):
        t = t[:-1]
    if re.fullmatch(r"-?\d+", t):
        return Fraction(int(t))
    if re.fullmatch(r"-?(\d+\.\d+|\.\d+)", t):
        return Fraction(t)
    m = re.fullmatch(r"(-?)(\d+)/(\d+)", t)
    if not m:
        m = re.fullmatch(r"(-?)\\d?frac\{(\d+)\}\{(\d+)\}", t)
    if m:
        a, b = int(m.group(2)), int(m.group(3))
        if b == 0:
            return None
        val = Fraction(a, b)
        return -val if m.group(1) else val
    return None

def accuracy_reward(response, reference):
    """1.0 if the extracted answer matches the reference (as numbers, else as whitespace-free lower-case strings)."""
    ans = extract_answer(response)
    if ans is None:
        return 0.0
    na, nr = to_number(ans), to_number(reference)
    if na is not None and nr is not None:
        return 1.0 if na == nr else 0.0
    norm = lambda s: re.sub(r"\s+", "", s).lower()
    return 1.0 if norm(ans) == norm(reference) else 0.0


def rule_reward(response, reference):
    """accuracy_reward + format_reward."""
    return float(accuracy_reward(response, reference) + format_reward(response))
