#!/usr/bin/env python3

import importlib.metadata
import logging
import os
import typer

from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from rsciio.image import file_writer as image_file_writer
from rsciio.tiff import file_reader as tiff_file_reader
from tqdm import tqdm
from tsio import __app_name__
from typing import Dict, List, Optional, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)

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


def write_page(config: Tuple[int, Dict, Path, FileFormats]) -> Path:
    index, page, destination, output_format = config
    LOGGER.debug(f"index={index}")
    LOGGER.debug(f"destination={destination}")
    LOGGER.debug(f"output_format={output_format}")
    output_file = destination.joinpath(str(index)).with_suffix(output_format.file_ext)
    image_file_writer(output_file, page)
    return output_file


@app.command(help="Handle Input/Output (IO) of TIFF files.")
def tiff(
    output_format: FileFormats = typer.Argument(help="The output file format."),
    files: List[Path] = typer.Argument(help="The original TIFF source files."),
    num_cpus: Optional[int] = typer.Option(None, "-n", "--num-cpus", help="The number of CPU cores to use for parallel execution."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Destination for output file(s)."),
    silent: bool = typer.Option(False, "-S", "--silent", help="Disables the progress bars.")
):
    LOGGER.debug(f"files={files}")
    LOGGER.debug(f"num_cpus={num_cpus}")
    LOGGER.debug(f"output={output}")
    LOGGER.debug(f"output_format={output_format}")
    for src in files:
        if output is None:
            destination = src.resolve().parent
        else:
            destination = output.resolve()
        src_file_stem = src.stem
        LOGGER.debug(f"src_file_stem={src_file_stem}")
        pages = tiff_file_reader(src, multipage_as_list=True)
        LOGGER.debug(f"len(pages)={len(pages)}")
        if len(pages) > 1:
            destination = destination.joinpath(src_file_stem)
            os.makedirs(destination, exist_ok=True)
        pages_list = [(index, page, destination, output_format) for index, page in enumerate(pages)]
        with Pool(num_cpus) as pool:
            list(tqdm(pool.imap(write_page, pages_list), total=len(pages), desc=src.name, disable=silent))


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
    LOGGER.debug(f"version={version}")


if __name__ == "__main__":
    app(prog_name=__app_name__)
