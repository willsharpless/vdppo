from typing import Any, NamedTuple


class EnvStep(NamedTuple):
    envstate: Any
    predicates: dict
    term: bool
    trunc: bool
    info: dict
