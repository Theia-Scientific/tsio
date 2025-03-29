#!/usr/bin/env python3

import importlib.metadata
import logging
import os
import typer

from enum import Enum
from pathlib import Path
from rsciio.image import file_writer as image_file_writer
from rsciio.tiff import file_reader as tiff_file_reader
from tsio import __app_name__
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)

PREFIX: str = f"{__app_name__.upper()}"

JPEG_FILE_EXT: str = ".jpg"
JPEG_MIME_TYPE: str = "image/jpeg"
PNG_FILE_EXT: str = ".png"
PNG_MIME_TYPE: str = "image/png"
TIFF_FILE_EXT: str = ".tif"
TIFF_MIME_TYPE: str = "image/tiff"

app = typer.Typer(pretty_exceptions_show_locals=False)

class UnsupportedFileFormat(Exception):
    def __init__(self, value: str):
        self.value = value


class FileFormats(Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

    @property
    def mime_type(self) -> str:
        MIME_TYPES = {
            FileFormats.JPEG: JPEG_MIME_TYPE,
            FileFormats.PNG: PNG_MIME_TYPE,
            FileFormats.TIFF: TIFF_MIME_TYPE,
        }
        mime_type = MIME_TYPES.get(self)
        if mime_type is None:
            raise UnsupportedFileFormat(self.value)
        return mime_type

    @property
    def file_ext(self) -> str:
        FILE_EXTS = {
            FileFormats.JPEG: JPEG_FILE_EXT,
            FileFormats.PNG: PNG_FILE_EXT,
            FileFormats.TIFF: TIFF_FILE_EXT,
        }
        file_ext = FILE_EXTS.get(self)
        if file_ext is None:
            raise UnsupportedFileFormat(self.value)
        return file_ext


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
def tiff(
    output_format: FileFormats = typer.Argument(help="The output file format."),
    input_path: Path = typer.Argument(help="The input file."),
    output: Optional[Path] = typer.Option(None, help="Destination for output file(s).")
):
    logger.debug(f"input_path={input_path}")
    logger.debug(f"output={output}")
    logger.debug(f"output_format={output_format}")
    if output is None:
        destination = input_path.resolve().parent
    else:
        destination = output.resolve()
    input_file_stem = input_path.stem
    logger.debug(f"input_file_stem={input_file_stem}")
    pages = tiff_file_reader(input_path, multipage_as_list=True)
    logger.debug(f"len(pages)={len(pages)}")
    if len(pages) > 1:
        destination = destination.joinpath(input_file_stem)
        os.makedirs(destination, exist_ok=True)
    for index, page in enumerate(pages):
        logger.debug(f"index={index}")
        output_file = destination.joinpath(str(index)).with_suffix(output_format.file_ext)
        logger.info(f"Writing '{output_file}'...")
        image_file_writer(output_file, page)
        logger.info(f"Writing '{output_file}'...DONE")


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
