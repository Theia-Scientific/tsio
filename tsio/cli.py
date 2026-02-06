#!/usr/bin/env python3

import cv2
import importlib.metadata
import logging
import mimetypes
import numpy as np
import os
import typer

from enum import Enum
from multiprocessing import Pool
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

    
def write_dataset(config: Tuple[int, Dict, Path, OutputFileFormats]) -> Path:
    index, dataset, destination, output_format = config
    LOGGER.debug(f"index={index}")
    LOGGER.debug(f"destination={destination}")
    LOGGER.debug(f"output_format={output_format}")
    output_file = destination.joinpath(str(index)).with_suffix(output_format.file_ext)
    src = dataset["data"]
    normalized_image = ((src - np.min(src)) / (np.max(src) - np.min(src))).astype(
        np.float32
    )
    img = cv2.cvtColor(
        np.round(normalized_image * 256).astype(BIT_DEPTH_DTYPE), cv2.COLOR_GRAY2BGR
    )
    image_file_writer(output_file, img)
    return output_file


def write_page(config: Tuple[int, Dict, Path, OutputFileFormats]) -> Path:
    index, page, destination, output_format = config
    LOGGER.debug(f"index={index}")
    LOGGER.debug(f"destination={destination}")
    LOGGER.debug(f"output_format={output_format}")
    output_file = destination.joinpath(str(index)).with_suffix(output_format.file_ext)
    image_file_writer(output_file, page)
    return output_file


@app.command(help="Handle Input/Output (IO) of DM3/4 files.")
def dm4(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    files: List[Path] = typer.Argument(help="The original DM3/4 source files."),
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
        datasets = dm_file_reader(src)
        LOGGER.debug(f"len(pages)={len(datasets)}")
        if len(datasets) > 1:
            destination = destination.joinpath(src_file_stem)
            os.makedirs(destination, exist_ok=True)
        datasets_list = [(index, dataset, destination, output_format) for index, dataset in enumerate(datasets)]
        with Pool(num_cpus) as pool:
            list(tqdm(pool.imap(write_dataset, datasets_list), total=len(datasets), desc=src.name, disable=silent))


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
