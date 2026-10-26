import pytest

from engrag.tools import calculate, UnsafeExpressionError


def test_calculate_basic_arithmetic():
    assert calculate("2 + 3") == 5
    assert calculate("480 * 520") == 249600
    assert calculate("10 / 4") == 2.5
    assert calculate("2 ** 10") == 1024


def test_calculate_respects_operator_precedence():
    assert calculate("2 + 3 * 4") == 14
    assert calculate("(2 + 3) * 4") == 20


def test_calculate_rejects_name_lookups():
    with pytest.raises(UnsafeExpressionError):
        calculate("__import__('os').system('echo hi')")


def test_calculate_rejects_function_calls():
    with pytest.raises(UnsafeExpressionError):
        calculate("abs(-5)")


def test_calculate_rejects_garbage_syntax():
    with pytest.raises(UnsafeExpressionError):
        calculate("not an expression")
