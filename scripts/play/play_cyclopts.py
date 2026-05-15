from pathlib import Path
from loguru import logger
from typing import Annotated

import cyclopts.config
import ipdb
from cyclopts import App, Argument, ArgumentCollection, Parameter, Token

app = App()


class DerivedDefaults:
    def __call__(self, app: App, commands: tuple[str, ...], arguments: ArgumentCollection):
        # if "url" in arguments:
        #
        # d = cyclopts.config.Dict(d)
        logger.debug("arguments: {}".format(arguments))
        # argument = Argument()
        # arguments.append(argument)


@app.command
def run(*, env: str = "dev", url: str | None = None):
    print(env, url)


@app.meta.default
def meta(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    config: Path = Path("pyproject.toml"),
):
    logger.debug("in meta")
    app.config = [
        cyclopts.config.Toml(config, root_keys=["tool", "myapp"], search_parents=True),
    ]
    app(tokens)


if __name__ == "__main__":
    app.meta()
