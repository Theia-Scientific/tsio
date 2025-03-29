#!/usr/bin/env python3

import importlib.metadata
import logging
import typer

from tsio import __app_name__
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)

PREFIX: str = f"{__app_name__.upper()}"
TIFF_MIME_TYPE: str = "image/tiff"

app = typer.Typer(pretty_exceptions_show_locals=False)

def map_verbosity(enabled: bool) -> str:
    if enabled:
        return "DEBUG"
    else:
        return "INFO"


def version_callback(value: bool):
    if value:
        version = importlib.metadata.version(__app_name__)
        print(f"{__app_name__} {version}")
        raise typer.Exit()


@app.command(help="Handle Input/Output (IO) of TIFF files.")
def tiff():
    pass


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print debugging statements.",
        envvar=f"{PREFIX}_VERBOSE",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Prints the version.",
        callback=version_callback,
        is_eager=True,
    ),
):
    logging.basicConfig(level=map_verbosity(verbose))
    logger.debug(f"version={version}")


if __name__ == "__main__":
    app(prog_name=__app_name__)
