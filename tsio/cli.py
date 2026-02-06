#!/usr/bin/env python3

import cv2
import importlib.metadata
import logging
import mimetypes
import numpy as np
import os
import typer

from enum import Enum
from multiprocess.pool import Pool
from pathlib import Path
from rsciio.digitalmicrograph import file_reader as dm_file_reader
from rsciio.image import file_writer as image_file_writer
from rsciio.tiff import file_reader as tiff_file_reader
from tqdm import tqdm
from tsio import __app_name__
from typing import Dict, List, Optional, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)
PREFIX: str = f"{__app_name__.upper()}"

BIT_DEPTH_DTYPE: str = "uint8"
DM4_FILE_EXT: str = ".dm4"
DM3_FILE_EXT: str = ".dm3"
DM3_MIME_TYPE: str = "application/vnd.gatan.dm3"
DM4_MIME_TYPE: str = "application/vnd.gatan.dm4"
JPEG_FILE_EXT: str = ".jpg"
JPEG_MIME_TYPE: str = "image/jpeg"
PNG_FILE_EXT: str = ".png"
PNG_MIME_TYPE: str = "image/png"
TIFF_FILE_EXT: str = ".tif"
TIFF_MIME_TYPE: str = "image/tiff"

mimetypes.add_type(DM3_MIME_TYPE, DM3_FILE_EXT)
mimetypes.add_type(DM4_MIME_TYPE, DM4_FILE_EXT)

logging.getLogger("PIL.Image").setLevel(logging.WARNING)

app = typer.Typer(pretty_exceptions_show_locals=False)

class UnsupportedFileFormat(Exception):
    def __init__(self, value: str):
        self.value = value


class OutputFileFormats(Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

    @property
    def mime_type(self) -> str:
        MIME_TYPES = {
            OutputFileFormats.JPEG: JPEG_MIME_TYPE,
            OutputFileFormats.PNG: PNG_MIME_TYPE,
            OutputFileFormats.TIFF: TIFF_MIME_TYPE,
        }
        mime_type = MIME_TYPES.get(self)
        if mime_type is None:
            raise UnsupportedFileFormat(self.value)
        return mime_type

    @property
    def file_ext(self) -> str:
        FILE_EXTS = {
            OutputFileFormats.JPEG: JPEG_FILE_EXT,
            OutputFileFormats.PNG: PNG_FILE_EXT,
            OutputFileFormats.TIFF: TIFF_FILE_EXT,
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

    
def write_dataset(config: Tuple[int, Dict, Path, OutputFileFormats, Optional[str]]) -> Path:
    index, dataset, destination, output_format, src_file_stem = config
    LOGGER.debug(f"{index=}")
    LOGGER.debug(f"{destination=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{src_file_stem=}")
    if src_file_stem is None:
        output_file = destination.joinpath(str(index)).with_suffix(output_format.file_ext)
    else:
        output_file = destination.joinpath(src_file_stem).with_suffix(output_format.file_ext)
    LOGGER.debug(f"{output_file=}")
    src = dataset["data"]
    normalized_image = ((src - np.min(src)) / (np.max(src) - np.min(src))).astype(
        np.float32
    )
    img = cv2.cvtColor(
        np.round(normalized_image * 256).astype(BIT_DEPTH_DTYPE), cv2.COLOR_GRAY2BGR
    )
    dataset["data"] = img
    image_file_writer(output_file, dataset)
    return output_file


def write_page(config: Tuple[int, Dict, Path, OutputFileFormats, Optional[str]]) -> Path:
    index, page, destination, output_format, src_file_stem = config
    LOGGER.debug(f"{index=}")
    LOGGER.debug(f"{destination=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{src_file_stem=}")
    if src_file_stem is None:
        output_file = destination.joinpath(str(index)).with_suffix(output_format.file_ext)
    else:
        output_file = destination.joinpath(src_file_stem).with_suffix(output_format.file_ext)
    LOGGER.debug(f"{output_file=}")
    image_file_writer(output_file, page)
    return output_file


def write_dm(config: Tuple[Path, Optional[Path], OutputFileFormats, Optional[int], bool]):
    src, output, output_format, num_cpus, silent = config
    LOGGER.debug(f"{src=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{silent=}")
    if output is None:
        destination = src.resolve().parent
    else:
        destination = output.resolve()
    src_file_stem = src.stem
    LOGGER.debug(f"{src_file_stem=}")
    try:
        datasets = dm_file_reader(src)
        LOGGER.debug(f"len(datasets)={len(datasets)}")
        if len(datasets) > 1:
            destination = destination.joinpath(src_file_stem)
            os.makedirs(destination, exist_ok=True)
            src_file_stem = None
        datasets_list = [(index, dataset, destination, output_format, src_file_stem) for index, dataset in enumerate(datasets)]
        for dataset in list(tqdm(datasets_list, total=len(datasets), desc=src.name, disable=silent)):
            write_dataset(dataset)
    except NotImplementedError as error:
        LOGGER.warning(f"Skipped '{src}' bceause: '{str(error)}'")
    except Exception as error:
        LOGGER.error(f"Skipped '{src}' because: '{str(error)}'")


@app.command(help="Handle Input/Output (IO) of DigitalMicrograph (DM) files.")
def dm(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original DM source files."),
    num_cpus: Optional[int] = typer.Option(None, "-n", "--num-cpus", help="The number of CPU cores to use for parallel execution."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Destination for output file(s)."),
    silent: bool = typer.Option(False, "-S", "--silent", help="Disables the progress bars.")
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    sources = []
    for path in paths:
        if path.is_dir():
            sources.extend([(path.joinpath(p), output, output_format, num_cpus, silent) for p in os.listdir(path) if path.joinpath(p).is_file()])
        else:
            sources.append((path, output, output_format, num_cpus, silent))
    with Pool(num_cpus) as pool:
        list(pool.imap(write_dm, sources))


@app.command(help="Handle Input/Output (IO) of TIFF files.")
def tiff(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
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
            src_file_stem = None
        pages_list = [(index, page, destination, output_format, src_file_stem) for index, page in enumerate(pages)]
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
