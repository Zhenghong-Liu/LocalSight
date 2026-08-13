"""工具执行器：6 种内置工具 + <tool_call> 解析。"""
from __future__ import annotations

import ast
import json
import operator
import re
import time
from typing import Any

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_SAFE_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_EXCHANGE_RATES = {"USD_CNY": 7.21, "CNY_USD": 1 / 7.21, "USD_EUR": 0.92, "EUR_USD": 1 / 0.92,
                   "CNY_EUR": 0.128, "EUR_CNY": 1 / 0.128}
_WEATHER = {"北京": "晴，26°C", "上海": "多云，28°C", "广州": "雷阵雨，30°C",
            "深圳": "多云，29°C", "杭州": "小雨，24°C", "成都": "阴，22°C"}
_TRANSLATE = {"你好": "hello", "谢谢": "thank you", "世界": "world", "hello": "你好"}
_UNITS = {"km_m": 1000, "m_km": 1 / 1000, "kg_g": 1000, "g_kg": 1 / 1000,
          "h_min": 60, "min_h": 1 / 60, "c_f": lambda x: x * 9 / 5 + 32, "f_c": lambda x: (x - 32) * 5 / 9}


class ToolError(Exception):
    """工具执行错误（奖励函数据此给低分，而不是训练崩溃）。"""


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](visit(node.operand))
        raise ToolError(f"不支持的表达式: {expr}")

    try:
        return visit(tree)
    except ZeroDivisionError as exc:
        raise ToolError("除零错误") from exc


def calculate_math(expression: str) -> str:
    return json.dumps({"result": _safe_eval(str(expression))}, ensure_ascii=False)


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    key = f"{from_currency.upper()}_{to_currency.upper()}"
    if key not in _EXCHANGE_RATES:
        raise ToolError(f"不支持的货币对: {key}")
    return json.dumps({"from": from_currency, "to": to_currency, "rate": _EXCHANGE_RATES[key]}, ensure_ascii=False)


def get_current_weather(city: str) -> str:
    return json.dumps({"city": city, "weather": _WEATHER.get(city, "晴，25°C"), "source": "stub"}, ensure_ascii=False)


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    return json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}, ensure_ascii=False)


def translate_text(text: str, target_language: str = "zh") -> str:
    result = _TRANSLATE.get(text.strip().lower())
    if result is None:
        raise ToolError(f"词表未覆盖: {text}")
    return json.dumps({"text": text, "translation": result, "target_language": target_language}, ensure_ascii=False)


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    key = f"{from_unit}_{to_unit}"
    if key not in _UNITS:
        raise ToolError(f"不支持的换算: {key}")
    factor = _UNITS[key]
    result = factor(value) if callable(factor) else value * factor
    return json.dumps({"value": result, "from": from_unit, "to": to_unit}, ensure_ascii=False)


TOOL_REGISTRY = {
    "calculate_math": calculate_math,
    "get_exchange_rate": get_exchange_rate,
    "get_current_weather": get_current_weather,
    "get_current_time": get_current_time,
    "translate_text": translate_text,
    "unit_converter": unit_converter,
}


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """解析 <tool_call>{json}</tool_call>（兼容 {'name':..., 'arguments':...} 或 {'function':...}）。"""
    calls: list[tuple[str, dict]] = []
    for raw in _CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        fn = obj.get("function", obj)
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and isinstance(args, dict):
            calls.append((name, args))
    return calls


def execute_tool(name: str, arguments: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"未知工具: {name}")
    return fn(**arguments)
