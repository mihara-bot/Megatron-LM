# Derived from pretrain_LLM_pipeline src/eval/math_grader.py (commit 1aa819b6): the same grading, so pass@k is
# comparable across the two code bases. Self-contained; this copy is the ladder's own.
"""Answer extraction and grading for GSM8K and MATH, plus the unbiased pass@k estimator.

MATH follows the Minerva protocol (Lewkowycz et al. 2022, Appendix D; the version in lm-eval's
``minerva_math``): take the last ``\\boxed{...}``, normalize it with :func:`normalize_final_answer`,
and count it correct if it equals the normalized reference or sympy finds the two LaTeX
expressions equal (``parse_latex``, difference simplifies to 0, 5 s timeout). sympy's LaTeX parser
needs ``antlr4-python3-runtime``; without it, grading falls back to the normalized string match and
:func:`sympy_available` reports False.

GSM8K: the answer is the last number inside the last ``\\boxed{}``, or the last number of the
completion when there is no box, compared numerically with the number after ``####``.

Countdown (TinyZero rule): the last ``\\boxed{}`` holds an equation; after turning LaTeX operators
into ``+ - * /`` it may contain only numbers, those operators and parentheses, must use every given
number exactly once, and must evaluate (exactly, with fractions) to the target. Fallback: a box
holding only the target value counts when the solution text writes a valid equation
(``fallback: True`` in the record; see :func:`grade_countdown`).

Standard library only; sympy is imported lazily.
"""

from __future__ import annotations

import ast
import json
import math
import re
import signal
import threading
from contextlib import contextmanager
from fractions import Fraction
from typing import Iterator, Optional

# --------------------------------------------------------------------------------------------
# Boxed answers (Hendrycks et al. MATH utilities)
# --------------------------------------------------------------------------------------------


def last_boxed_only_string(text: str) -> Optional[str]:
    """The last ``\\boxed{...}`` (or ``\\fbox{...}``) with balanced braces, or None."""
    idx = text.rfind("\\boxed")
    if idx >= 0 and text.startswith("\\boxed ", idx):  # "\boxed 5" form
        return "\\boxed " + text[idx + len("\\boxed ") :].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return None


def remove_boxed(boxed: str) -> str:
    if boxed.startswith("\\boxed "):
        return boxed[len("\\boxed ") :]
    for left in ("\\boxed{", "\\fbox{"):
        if boxed.startswith(left) and boxed.endswith("}"):
            return boxed[len(left) : -1]
    return boxed


def extract_boxed(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    return None if boxed is None else remove_boxed(boxed)


# --------------------------------------------------------------------------------------------
# Minerva normalization
# --------------------------------------------------------------------------------------------

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm",
    "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Minerva's normalization of a final answer string (Lewkowycz et al. 2022, Listing 1).

    One addition: ``\\dfrac`` / ``\\tfrac`` become ``\\frac`` first, so display-style fractions
    match the references (MATH golds mostly use ``\\frac``).
    """
    final_answer = final_answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract the answer that is in LaTeX math, bold, boxed, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Shorthand TeX: \fracab -> \frac{a}{b}, \sqrta -> \sqrt{a}.
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


# --------------------------------------------------------------------------------------------
# sympy equivalence
# --------------------------------------------------------------------------------------------

_SYMPY_STATE = {"checked": False, "ok": False}


def sympy_available() -> bool:
    """Whether sympy's LaTeX parser works here (needs antlr4-python3-runtime)."""
    if not _SYMPY_STATE["checked"]:
        _SYMPY_STATE["checked"] = True
        try:
            from sympy.parsing.latex import parse_latex

            parse_latex("1+1")
            _SYMPY_STATE["ok"] = True
        except Exception:  # ImportError, or antlr runtime missing / mismatched
            _SYMPY_STATE["ok"] = False
    return _SYMPY_STATE["ok"]


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """SIGALRM timeout in the main thread; no limit elsewhere."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError()

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def sympy_equal(a: str, b: str, timeout_seconds: int = 5) -> bool:
    """Minerva ``is_equiv``: parse both as LaTeX and check that the difference simplifies to 0."""
    if not sympy_available():
        return False
    import sympy
    from sympy.parsing.latex import parse_latex

    try:
        with _time_limit(timeout_seconds):
            try:
                diff = parse_latex(a) - parse_latex(b)
            except Exception:
                return False
            try:
                return bool(sympy.simplify(diff) == 0)
            except Exception:
                return False
    except TimeoutError:
        return False


# --------------------------------------------------------------------------------------------
# Task graders
# --------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _last_number(text: str) -> Optional[float]:
    found = _NUMBER.findall(text.replace(",", ""))
    if not found:
        return None
    try:
        return float(found[-1])
    except ValueError:
        return None


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def grade_gsm8k(completion: str, gold: str) -> dict:
    boxed = extract_boxed(completion)
    pred = _last_number(boxed) if boxed is not None else _last_number(completion)
    target = _last_number(gold)
    correct = pred is not None and target is not None and math.isclose(pred, target, rel_tol=0.0, abs_tol=1e-6)
    return {"extracted": boxed if boxed is not None else (None if pred is None else repr(pred)),
            "boxed": boxed is not None, "correct": bool(correct)}


def grade_math(completion: str, gold: str) -> dict:
    boxed = extract_boxed(completion)
    if boxed is None:
        return {"extracted": None, "boxed": False, "correct": False}
    pred = normalize_final_answer(boxed)
    target = normalize_final_answer(gold)
    correct = pred == target or _numbers_equal(pred, target) or sympy_equal(pred, target)
    return {"extracted": pred, "boxed": True, "correct": bool(correct)}


def _plain_number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _numbers_equal(a: str, b: str) -> bool:
    """Both plain numbers and equal ("27" vs "27.0"); what sympy would also accept, without antlr."""
    x, y = _plain_number(a), _plain_number(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)


# --------------------------------------------------------------------------------------------
# Countdown
# --------------------------------------------------------------------------------------------

_LATEX_OPS = (
    ("\\left", ""), ("\\right", ""), ("\\times", "*"), ("\\cdot", "*"), ("\\ast", "*"), ("\\div", "/"),
    ("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"), ("$", ""), ("\\,", ""), ("\\!", ""), ("\\ ", ""),
)
_FRAC = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_COUNTDOWN_CHARS = re.compile(r"^[\d+\-*/().\s]+$")
_HAS_OPERATOR = re.compile(r"[+\-*/]")


def countdown_expression(boxed: str) -> str:
    """Plain arithmetic from a boxed answer: LaTeX operators and \\frac replaced, the
    ``= target`` side dropped (the first side that contains an operator is kept)."""
    expr = boxed
    for before, after in _LATEX_OPS:
        expr = expr.replace(before, after)
    while True:
        replaced = _FRAC.sub(r"((\1)/(\2))", expr)
        if replaced == expr:
            break
        expr = replaced
    expr = expr.replace("{", "(").replace("}", ")")
    sides = [side.strip() for side in expr.split("=")]
    with_op = [side for side in sides if _HAS_OPERATOR.search(side)]
    return with_op[0] if with_op else sides[0]


def _eval_arithmetic(expr: str) -> Fraction:
    """Exact value of an expression of numbers, + - * / and parentheses (no eval())."""

    def ev(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = ev(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        raise ValueError(f"not plain arithmetic: {ast.dump(node)}")

    return ev(ast.parse(expr, mode="eval"))


def countdown_gold(nums, target) -> str:
    return json.dumps({"nums": [int(n) for n in nums], "target": int(target)}, sort_keys=True)


def _countdown_valid(expr: str, spec: dict) -> bool:
    """Plain arithmetic using every given number exactly once and equal to the target."""
    if not expr or not _COUNTDOWN_CHARS.match(expr) or len(expr) > 200:
        return False
    used = sorted(int(n) for n in re.findall(r"\d+", expr))
    if used != sorted(int(n) for n in spec["nums"]):
        return False
    try:
        return _eval_arithmetic(expr) == Fraction(int(spec["target"]))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return False


# Left-hand sides of "... = ..." in running text: a run of arithmetic characters on one line.
_TEXT_EQUATION_LHS = re.compile(r"([\d+\-*/(). \t]+)=")


def _plain_text_arithmetic(text: str) -> str:
    for before, after in _LATEX_OPS:
        text = text.replace(before, after)
    while True:
        replaced = _FRAC.sub(r"((\1)/(\2))", text)
        if replaced == text:
            return text
        text = replaced


def countdown_text_equation(completion: str, spec: dict) -> Optional[str]:
    """The last valid equation (left of an ``=``) written in the solution text, or None."""
    found = None
    for m in _TEXT_EQUATION_LHS.finditer(_plain_text_arithmetic(completion)):
        expr = m.group(1).strip()
        if _HAS_OPERATOR.search(expr) and _countdown_valid(expr, spec):
            found = expr
    return found


def grade_countdown(completion: str, gold: str) -> dict:
    """``gold`` is JSON ``{"nums": [...], "target": t}`` (see :func:`countdown_gold`).

    Correct when the last ``\\boxed{}`` holds a valid equation (TinyZero rule). Fallback: when the
    box holds just the target value (an answer-format slip; the SFT data always boxes a value), the
    last valid equation written in the solution text counts, and the record has ``fallback: True``.
    The summary reports ``strict_pass@1`` without these and ``fallback_rate``.
    """
    boxed = extract_boxed(completion)
    if boxed is None:
        return {"extracted": None, "boxed": False, "correct": False, "fallback": False}
    spec = json.loads(gold)
    expr = countdown_expression(boxed)
    if _countdown_valid(expr, spec):
        return {"extracted": expr, "boxed": True, "correct": True, "fallback": False}
    if not _HAS_OPERATOR.search(expr.lstrip("-")):
        try:
            value_is_target = Fraction(expr.strip()) == Fraction(int(spec["target"]))
        except (ValueError, ZeroDivisionError):
            value_is_target = False
        if value_is_target:
            equation = countdown_text_equation(completion, spec)
            if equation is not None:
                return {"extracted": equation, "boxed": True, "correct": True, "fallback": True}
    return {"extracted": expr, "boxed": True, "correct": False, "fallback": False}


GRADERS = {"gsm8k": grade_gsm8k, "math": grade_math, "countdown": grade_countdown}


# --------------------------------------------------------------------------------------------
# pass@k
# --------------------------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``n`` samples with ``c`` correct (Chen et al. 2021): 1 - C(n-c,k)/C(n,k)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n; got n={n}, c={c}, k={k}.")
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_wrong *= 1.0 - k / i
    return 1.0 - prob_all_wrong


def pass_k_values(num_samples: int) -> list:
    """k = 1, 2, 4, ... up to num_samples, plus num_samples itself."""
    ks = []
    k = 1
    while k <= num_samples:
        ks.append(k)
        k *= 2
    if num_samples not in ks:
        ks.append(num_samples)
    return ks
