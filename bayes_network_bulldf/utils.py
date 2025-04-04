from typing import Any, Sequence


def str_to_value(value: str) -> Any:
    if value.isdecimal() or value in {'True', 'False'}:
        return eval(value)
    return value

def strs_to_domain(values: Sequence[str]) -> set:
    ans = set()
    for value in values:
        ans.add(str_to_value(value))
    return set([str_to_value(value) for value in values])

def dict_to_frozenset(dictionary: dict[str, Any]) -> frozenset[str]:
    return frozenset([f'{key}: {value}' for key, value in dictionary.items()])