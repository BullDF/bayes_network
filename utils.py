from typing import Any

def str_to_value(value: str) -> Any:
    if value.isdecimal():
        return int(value)
    elif value in {True, False}:
        return bool(value)
    return value