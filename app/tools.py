"""Deterministic tools the agent can call instead of letting the LLM do arithmetic.

This is the honest fix for the thing every edtech RAG bot gets wrong: language
models are unreliable at numeric work, so numeric work is routed to SymPy and
the model only narrates the result.
"""

from __future__ import annotations

import multiprocessing as mp
import re
from functools import partial

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.config import TOOL_MEMORY_MB, TOOL_TIMEOUT

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_ALLOWED = {name: getattr(sp, name) for name in (
    "pi", "E", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "log", "exp", "Abs", "Rational", "Symbol",
)}

# IMPORTANT: sympy's parse_expr ultimately calls eval(). On its own it will
# happily evaluate __import__('os').system('rm -rf /'), because Python injects
# real builtins into any globals mapping that lacks __builtins__. Two guards:
#   1. a character/identifier allowlist applied to the raw string, and
#   2. an empty __builtins__ in the globals handed to the parser.
# Both are needed; the regex alone can be out-thought, and the empty builtins
# alone still permits sympy internals to be reached by name.
_SAFE_CHARS = re.compile(r"^[0-9a-zA-Z+\-*/^().,\s=<>!]*$")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_ATTRIBUTE = re.compile(r"\.\s*[A-Za-z]")
# 3e8 is a number, not the name e8. The boundaries stop this from hiding a real
# name from the checks below, as it would in x3e8 or 3e8.n.
_SCI_NUMBER = re.compile(r"(?<![A-Za-z0-9.])(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+(?![A-Za-z0-9.])")
_SAFE_NAMES = set(_ALLOWED) | {
    # single/short symbols a physics problem legitimately uses as variables
    *"abcdefghijklmnopqrstuvwxyz",
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
}
# parse_expr's tokenizer rewrites literals as Integer(2) / Float(9.8) / Symbol('v'),
# so those constructors must exist in globals even though no user may name them
# (the identifier allowlist above rejects them in raw input).
_CONSTRUCTORS = {
    name: getattr(sp, name)
    for name in ("Integer", "Float", "Rational", "Symbol", "Add", "Mul", "Pow")
}
_GLOBALS = {"__builtins__": {}, **_ALLOWED, **_CONSTRUCTORS}

MAX_EXPR_LEN = 300


class UnsafeExpression(ValueError):
    pass


def _guard(expression: str) -> str:
    expr = (expression or "").strip()
    if not expr:
        raise UnsafeExpression("empty expression")
    if len(expr) > MAX_EXPR_LEN:
        raise UnsafeExpression(f"expression longer than {MAX_EXPR_LEN} characters")
    if "_" in expr:
        raise UnsafeExpression("underscores are not allowed")
    if not _SAFE_CHARS.match(expr):
        raise UnsafeExpression("expression contains disallowed characters")
    bare = _SCI_NUMBER.sub("0", expr)
    if _ATTRIBUTE.search(bare):  # e.g. pi.n(5000): the dot is meant for decimals
        raise UnsafeExpression("attribute access is not allowed")
    for name in _IDENTIFIER.findall(bare):
        if name not in _SAFE_NAMES:
            raise UnsafeExpression(f"unknown name: {name}")
    return expr


def _parse(expression: str):
    return parse_expr(
        _guard(expression),
        local_dict=dict(_ALLOWED),
        global_dict=dict(_GLOBALS),
        transformations=_TRANSFORMS,
        evaluate=True,
    )


def _calculate(expression: str) -> dict:
    """Evaluate a numeric/symbolic expression. Returns exact and decimal forms."""
    try:
        expr = _parse(expression)
        exact = sp.simplify(expr)
        try:
            numeric = float(sp.N(exact))
        except (TypeError, ValueError):
            numeric = None
        return {
            "ok": True,
            "input": expression,
            "exact": str(exact),
            "value": numeric,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        return {"ok": False, "input": expression, "error": str(exc)}


def _solve_equation(equation: str, variable: str = "x") -> dict:
    """Solve 'lhs = rhs' (or an expression assumed equal to zero) for a variable."""
    try:
        _guard(variable)
        var = sp.Symbol(variable)
        if "=" in equation:
            lhs_raw, rhs_raw = equation.split("=", 1)
            eq = sp.Eq(_parse(lhs_raw), _parse(rhs_raw))
        else:
            eq = _parse(equation)
        roots = sp.solve(eq, var, dict=False)
        return {
            "ok": True,
            "input": equation,
            "variable": variable,
            "solutions": [str(r) for r in roots],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "input": equation, "error": str(exc)}


# A thread can't be killed, so 9^9^9 would hold a threadpool slot forever. Tools
# run in a child process that can be. forkserver keeps sympy preloaded on Linux;
# Windows only has spawn, which re-imports sympy per call.
_MP = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
if _MP.get_start_method() == "forkserver":
    _MP.set_forkserver_preload(["app.tools"])


def _limit_memory() -> None:
    try:
        import resource
    except ImportError:  # Windows: only the timeout applies
        return
    limit = TOOL_MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _worker(conn, fn, args: tuple, kwargs: dict) -> None:
    try:
        _limit_memory()
        conn.send(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - e.g. the model omitted an argument
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _bounded(fn, *args, **kwargs) -> dict:
    """Run a tool in a child process, killing it after TOOL_TIMEOUT seconds."""
    recv, send = _MP.Pipe(duplex=False)
    proc = _MP.Process(target=_worker, args=(send, fn, args, kwargs), daemon=True)
    proc.start()
    send.close()
    try:
        if recv.poll(TOOL_TIMEOUT):
            return recv.recv()
        return {"ok": False, "error": f"timed out after {TOOL_TIMEOUT:g}s"}
    finally:
        proc.kill()
        proc.join()
        recv.close()


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a numeric or symbolic physics expression. Use this for "
                "EVERY arithmetic step instead of computing it yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '0.5 * 2 * 9.8**2' or 'sqrt(2*9.8*20)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Solve an equation for a variable, e.g. 'v**2 = u**2 + 2*a*s'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string"},
                    "variable": {"type": "string", "default": "x"},
                },
                "required": ["equation"],
            },
        },
    },
]

calculate = partial(_bounded, _calculate)
solve_equation = partial(_bounded, _solve_equation)

TOOL_IMPLS = {"calculate": calculate, "solve_equation": solve_equation}
