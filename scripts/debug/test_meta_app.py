import pathlib
from typing import Annotated

import ipdb
from attrs import define
from cyclopts import App, Group, Parameter
from loguru import logger

app = App()
# app.help_flags = ["--help-new"]
app.meta.help_flags = ["--help-meta"]

app.meta.group_parameters = Group("Meta Parameters", sort_key=0)


@define
class MyCls:
    a: int = 3
    b: int = 4


@app.command()
def hi(x: MyCls = MyCls()):
    print(x)


@app.meta.default
def my_app_launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)], lol: MyCls = MyCls()
):
    logger.error(f"lol: {lol}")
    # Set the default argument of "loops" in the hi command to test.
    print(hi.__defaults__)
    hi.__defaults__ = (lol,)

    print(hi.__defaults__)

    app(tokens)


if __name__ == "__main__":
    app.meta()
