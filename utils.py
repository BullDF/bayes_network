from typing import Any

def str_to_value(value: str) -> Any:
    if value.isdecimal():
        return int(value)
    elif value in {True, False}:
        return bool(value)
    return value

def dict_to_frozenset(dictionary: dict[str, Any]) -> frozenset[str]:
    return frozenset([f'{key}: {value}' for key, value in dictionary.items()])