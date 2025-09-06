"""Tool registry available to agent."""

import json
import time
from typing import Any, Callable

from openai_harmony import ToolDescription


ToolFn = Callable[[dict[str, Any]], Any]


class Tools:
    """Registry of all tool functions."""
    REGISTRY: dict[str, tuple[ToolFn, str, dict[str, Any]]] = {}
    _INSTANCE = None

    def __init__(self):
        if Tools._INSTANCE is not None:
            return

        Tools._INSTANCE = self

    @classmethod
    def call(cls, name: str, args: dict[str, Any]) -> str:
        """Dispatch to a registered tool and always return JSON text."""
        if name not in Tools.REGISTRY:
            raise ValueError(f"Unknown tool: {name}")
        fn = Tools.REGISTRY[name][0]
        out = fn(args)
        return out if isinstance(out, str) else json.dumps(out)

    @classmethod
    def get_tool_descriptions(cls) -> list[ToolDescription]:
        """Return Harmony ToolDescription objects for all registered tools."""
        descs: list[ToolDescription] = []
        for name, (_, description, schema) in Tools.REGISTRY.items():
            descs.append(ToolDescription.new(name, description, parameters=schema))
        return descs

    @classmethod
    def register(cls, name: str, description: str, parameters: dict[str, Any]):
        """Decorator to register a tool function with metadata."""
        def deco(fn: ToolFn):
            Tools.REGISTRY[name] = (fn, description, parameters)
            return fn

        return deco

@Tools.register(
    name="get_current_weather",
    description="Gets the current weather in the provided location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and region, e.g., 'San Francisco, CA'."
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "fahrenheit"
            }
        },
        "required": ["location"]
    },
)
def get_current_weather(args: dict[str, Any]) -> dict[str, Any]:
    """Get the current weather for a location."""
    # Stub for now; swap with a real API later.
    loc = args.get("location", "Unknown")
    unit = args.get("format", "fahrenheit")
    temp_c = 20
    temp = temp_c if unit == "celsius" else (temp_c * 9 / 5) + 32
    return {"location": loc, "temperature": temp, "unit": unit, "sunny": True}

@Tools.register(
    name="get_time",
    description="Gets the current time in the provided location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and region, e.g., 'San Francisco, CA'."
            },
            "format": {
                "type": "string",
                "enum": ["12-hour", "24-hour"],
                "default": "12-hour"
            }
        },
        "required": ["location"]
    },
)
def get_time(args: dict[str, Any]) -> dict[str, Any]:
    """Get the current time for a location."""
    loc = args.get("location", "Unknown")
    fmt = args.get("format", "12-hour")
    curr_time = time.localtime()
    if fmt == "12-hour":
        time_str = time.strftime("%I:%M %p", curr_time)
    else:
        time_str = time.strftime("%H:%M", curr_time)
    return {"location": loc, "time": time_str, "format": fmt}

@Tools.register(
    name="get_stock_price",
    description="Gets the current stock price (USD) for a given company.",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock symbol of the company, e.g., 'AAPL' for Apple."
            }
        },
        "required": ["symbol"]
    },
)
def get_stock_price(args: dict[str, Any]) -> dict[str, Any]:
    """Get the current stock price for a company."""
    symbol = args.get("symbol", "Unknown")
    # Stub for now; swap with a real API later.
    return {"symbol": symbol, "price": 101.0}
