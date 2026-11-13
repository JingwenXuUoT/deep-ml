import ast
import math
import operator
import re

# Allowed operators for safe mathematical evaluation
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions and constants
FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "abs": abs,
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(expr: str) -> float:
  """Safely parses and evaluates a mathematical expression string."""
  # Clean whitespace and handle implicit multiplication (e.g., '2pi' -> '2*pi', '2(3)' -> '2*(3)')
  expr = expr.strip().replace("^", "**")
  expr = re.sub(r"(\d)\s*(pi|e|sqrt|\()", r"\1*\2", expr)
  expr = re.sub(r"(\))\s*(\d|pi|e|sqrt|\()", r"\1*\2", expr)

  def _eval(node):
    if isinstance(node, ast.Expression):
      return _eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float)
    ):
      return float(node.value)
    elif isinstance(node, ast.Name) and node.id in CONSTANTS:
      return CONSTANTS[node.id]
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
      return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
      return OPERATORS[type(node.op)](_eval(node.operand))
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
    ):
      args = [_eval(arg) for arg in node.args]
      return FUNCTIONS[node.func.id](*args)
    else:
      raise ValueError(f"Unsupported expression: {expr}")

  tree = ast.parse(expr, mode="eval")
  return _eval(tree)


def verify_math_answer(
    predicted: str, ground_truth: str, tolerance: float = 1e-6
) -> bool:
  """Verify if two mathematical answers are equivalent."""
  if predicted is None or ground_truth is None:
    return False

  pred_clean = predicted.strip()
  gt_clean = ground_truth.strip()

  # 1. Direct string equality
  if pred_clean == gt_clean:
    return True

  # 2. Numeric & expression equivalence
  try:
    pred_val = safe_eval(pred_clean)
    gt_val = safe_eval(gt_clean)

    # Handle NaN or Inf edge cases
    if math.isnan(pred_val) or math.isnan(gt_val):
      return False

    return math.isclose(pred_val, gt_val, rel_tol=tolerance, abs_tol=tolerance)
  except Exception:
    # Fallback to case-insensitive string match if neither is valid math
    return pred_clean.lower() == gt_clean.lower()
