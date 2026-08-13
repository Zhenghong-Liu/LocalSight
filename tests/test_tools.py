import pytest

from localsight.tools.executor import (
    ToolError,
    calculate_math,
    execute_tool,
    parse_tool_calls,
    unit_converter,
)


def test_calculate_math():
    assert calculate_math("2045*6994") == '{"result": 14302730}'
    with pytest.raises(ToolError):
        calculate_math("__import__('os').system('ls')")


def test_parse_tool_calls():
    text = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "1+1"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [("calculate_math", {"expression": "1+1"})]


def test_execute_tool_roundtrip():
    out = execute_tool("unit_converter", {"value": 1.5, "from_unit": "km", "to_unit": "m"})
    assert "1500.0" in out
    with pytest.raises(ToolError):
        execute_tool("no_such_tool", {})
