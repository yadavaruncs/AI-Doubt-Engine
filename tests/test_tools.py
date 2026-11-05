"""The calculator is the one component that must never be wrong or exploitable."""

import pytest

import sys
from types import SimpleNamespace as NS

import app.tools
from app.tools import calculate, solve_equation


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("sqrt(2*9.8*20)", 19.79898987322333),
        ("0.5 * 2 * 10**2", 100.0),
        ("9.8*3", 29.4),
        ("(1/2)*5*4^2", 40.0),   # caret is treated as a power, not XOR
        ("sin(pi/6)", 0.5),
        ("3e8*2", 6e8),
        ("3.e8", 3e8),
        ("2E3 + 1", 2001.0),
        ("6.626e-34*3e8", 1.9878e-25),
    ],
)
def test_calculate_numeric(expression, expected):
    result = calculate(expression)
    assert result["ok"], result
    assert result["value"] == pytest.approx(expected, rel=1e-6)


def test_calculate_symbolic_has_no_float_value():
    result = calculate("u + a*t")
    assert result["ok"]
    assert result["value"] is None
    assert "a*t" in result["exact"]


def test_solve_equation_quadratic():
    result = solve_equation("v**2 = 2*9.8*20", "v")
    assert result["ok"]
    roots = sorted(float(s) for s in result["solutions"])
    assert roots[1] == pytest.approx(19.798989, rel=1e-5)


def test_solve_equation_symbolic():
    result = solve_equation("u + a*t = 25", "t")
    assert result["ok"]
    assert result["solutions"] == ["(25 - u)/a"]


def test_malformed_expression_fails_cleanly():
    result = calculate("3 +* 4")
    assert not result["ok"]
    assert "error" in result


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "exec('import os')",
        "eval('1+1')",
        "lambda: 1",
        "[x for x in range(10)]",
        "Integer(2).__class__",
        "a" * 500,
        "pi.n(5000)",
        "3e8.n(5000)",
        "x3e8",
        "3e8x",
    ],
)
def test_parser_rejects_injection(payload):
    """sympy's parse_expr calls eval() underneath. Regression test for a real
    hole: without the guard in app.tools these payloads execute."""
    assert not calculate(payload)["ok"]


def test_decimals_still_allowed_after_attribute_guard():
    assert calculate("2.5 * .5")["value"] == pytest.approx(1.25)


def test_runaway_expression_is_killed(monkeypatch):
    monkeypatch.setattr(app.tools, "TOOL_TIMEOUT", 3)
    result = calculate("9^9^9")
    assert not result["ok"]
    assert "timed out" in result["error"]


def test_runaway_solve_is_killed_too(monkeypatch):
    monkeypatch.setattr(app.tools, "TOOL_TIMEOUT", 3)
    assert "timed out" in solve_equation("x = 9^9^9")["error"]


def test_wrong_tool_arguments_fail_cleanly():
    assert not calculate(wrong="1")["ok"]


def test_memory_limit_is_applied_where_supported(monkeypatch):
    calls = []
    fake = NS(RLIMIT_AS=9, setrlimit=lambda *a: calls.append(a))
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(app.tools, "TOOL_MEMORY_MB", 2)
    app.tools._limit_memory()
    assert calls == [(9, (2097152, 2097152))]
