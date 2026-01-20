import functools as ft

import cattrs
from attrs import astuple, define
from cattrs.strategies import configure_tagged_union, include_subclasses


@define(eq=False)
class Cfg:
    @staticmethod
    def get_converter():
        converter = cattrs.Converter()
        union_strategy = ft.partial(configure_tagged_union)
        include_subclasses(Cfg, converter, union_strategy=union_strategy)

        return converter

    @classmethod
    def fromdict(cls, d: dict, use_converter: bool = True):
        if use_converter:
            converter = Cfg.get_converter()
            return converter.structure(d, cls)

        return cattrs.structure(d, cls)

    def asdict(self) -> dict:
        converter = Cfg.get_converter()
        d = converter.unstructure(self)
        return d

    def astuple(self):
        return astuple(self)
