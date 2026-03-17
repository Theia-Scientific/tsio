#!/usr/bin/env python3

import cv2
import importlib.metadata
import logging
import mimetypes
import numpy as np
import os
import platform
import typer

from enum import Enum
from multiprocess.pool import Pool
from pathlib import Path
from pydicom import dcmread, iter_pixels
from rsciio import digitalmicrograph
from rsciio.image import (
    file_reader as image_file_reader,
    file_writer as image_file_writer,
)
from rsciio.tiff import file_reader as tiff_file_reader
from tqdm import tqdm
from tsio import __app_name__
from typing import Callable, Dict, List, Optional, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)
PREFIX: str = f"{__app_name__.upper()}"

BIT_DEPTH_DTYPE: str = "uint8"
DCM_FILE_EXT: str = ".dcm"
DCM_MIME_TYPE: str = "application/dicom"
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

mimetypes.add_type(DCM_MIME_TYPE, DCM_FILE_EXT)
mimetypes.add_type(DM3_MIME_TYPE, DM3_FILE_EXT)
mimetypes.add_type(DM4_MIME_TYPE, DM4_FILE_EXT)

logging.getLogger("PIL.Image").setLevel(logging.WARNING)

app = typer.Typer(pretty_exceptions_show_locals=False)


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
        return MIME_TYPES[self]

    @property
    def file_ext(self) -> str:
        FILE_EXTS = {
            OutputFileFormats.JPEG: JPEG_FILE_EXT,
            OutputFileFormats.PNG: PNG_FILE_EXT,
            OutputFileFormats.TIFF: TIFF_FILE_EXT,
        }
        return FILE_EXTS[self]

    @property
    def supported_bit_depths(self) -> List[str]:
        BIT_DEPTHS = {
            OutputFileFormats.JPEG: ["uint8"],
            OutputFileFormats.PNG: ["uint8", "uint16"],
            OutputFileFormats.TIFF: ["uint8", "uint16"],
        }
        return BIT_DEPTHS[self]


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


def write(
    pages: List[Dict],
    src: Path,
    output: Optional[Path],
    output_format: OutputFileFormats,
    silent: bool,
    normalize: bool = False,
):
    LOGGER.debug(f"{src=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    LOGGER.debug(f"{normalize=}")
    if output is None:
        destination = src.resolve().parent
    else:
        destination = output.resolve()
    pages_count = len(pages)
    LOGGER.debug(f"{pages_count}=")
    src_file_stem = src.stem
    LOGGER.debug(f"{src_file_stem=}")
    if pages_count > 1:
        destination = destination.joinpath(src_file_stem)
    os.makedirs(destination, exist_ok=True)
    LOGGER.debug(f"{src_file_stem=}")
    for page_index, page in enumerate(
        tqdm(pages, total=pages_count, desc=src.name, disable=silent)
    ):
        LOGGER.debug(f"{page_index=}")
        if pages_count > 1:
            output_file = destination.joinpath(str(page_index)).with_suffix(
                output_format.file_ext
            )
        else:
            output_file = destination.joinpath(src_file_stem).with_suffix(
                output_format.file_ext
            )
        LOGGER.debug(f"{output_file=}")
        img = page["data"]
        if normalize:
            img = ((img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.float32)
        LOGGER.debug(f"{img.dtype.name=}")
        if img.dtype.name not in output_format.supported_bit_depths:
            img_8bit = np.round(img * 256).astype(BIT_DEPTH_DTYPE)
            img = cv2.cvtColor(
                img_8bit,
                cv2.COLOR_GRAY2BGR,
            )
        page["data"] = img
        for axis in page["axes"]:
            if "navigate" not in axis:
                axis["navigate"] = None
        image_file_writer(output_file, page)


def write_dcm(config: Tuple[Path, Optional[Path], OutputFileFormats, bool]):
    src, output, output_format, silent = config
    write(
        [
            {
                "data": img,
                "axes": [],
                "index_in_array": None,
                "metadata": {},
                "original_metadata": {},
            }
            for img in iter_pixels(dcmread(src))
        ],
        src,
        output,
        output_format,
        silent,
        normalize=True,
    )


def write_dm(config: Tuple[Path, Optional[Path], OutputFileFormats, bool]):
    src, output, output_format, silent = config
    try:
        write(
            digitalmicrograph.file_reader(src),
            src,
            output,
            output_format,
            silent,
            normalize=True,
        )
    except NotImplementedError as error:
        LOGGER.warning(f"Skipped '{src}' because: '{str(error)}'")
    except Exception as error:
        LOGGER.error(f"Skipped '{src}' because: '{str(error)}'")


def write_png(config: Tuple[Path, Optional[Path], OutputFileFormats, bool]):
    src, output, output_format, silent = config
    write(
        image_file_reader(src),
        src,
        output,
        output_format,
        silent,
        normalize=False,
    )


def write_tiff(config: Tuple[Path, Optional[Path], OutputFileFormats, bool]):
    src, output, output_format, silent = config
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output,
        output_format,
        silent,
        normalize=False,
    )


def expand_sources(
    paths: List[Path],
    output: Optional[Path],
    output_format: OutputFileFormats,
    silent: bool,
) -> List[Tuple[Path, Optional[Path], OutputFileFormats, bool]]:
    sources = []
    for path in paths:
        if path.is_dir():
            sources.extend(
                [
                    (path.joinpath(p), output, output_format, silent)
                    for p in os.listdir(path)
                    if path.joinpath(p).is_file()
                ]
            )
        else:
            sources.append((path, output, output_format, silent))
    return sources


def run(
    write_func: Callable,
    sources: List[Tuple[Path, Optional[Path], OutputFileFormats, bool]],
    num_cpus: Optional[int] = None,
):
    if platform.system().lower() == "darwin":
        for src in sources:
            write_func(src)
    else:
        with Pool(num_cpus) as pool:
            list(pool.imap(write_func, sources))


@app.command(help="Handle Input/Output (IO) of DICOM (DCM) files.")
def dcm(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original DCM source files."),
    num_cpus: Optional[int] = typer.Option(
        None,
        "-n",
        "--num-cpus",
        help="The number of CPU cores to use for parallel execution.",
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Destination for output file(s)."
    ),
    silent: bool = typer.Option(
        False, "-S", "--silent", help="Disables the progress bars."
    ),
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(write_dcm, expand_sources(paths, output, output_format, silent), num_cpus)


@app.command(help="Handle Input/Output (IO) of DigitalMicrograph (DM) files.")
def dm(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original DM source files."),
    num_cpus: Optional[int] = typer.Option(
        None,
        "-n",
        "--num-cpus",
        help="The number of CPU cores to use for parallel execution.",
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Destination for output file(s)."
    ),
    silent: bool = typer.Option(
        False, "-S", "--silent", help="Disables the progress bars."
    ),
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(write_dm, expand_sources(paths, output, output_format, silent), num_cpus)


@app.command(help="Handle Input/Output (IO) of PNG files.")
def png(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original TIFF source files."),
    num_cpus: Optional[int] = typer.Option(
        None,
        "-n",
        "--num-cpus",
        help="The number of CPU cores to use for parallel execution.",
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Destination for output file(s)."
    ),
    silent: bool = typer.Option(
        False, "-S", "--silent", help="Disables the progress bars."
    ),
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(write_png, expand_sources(paths, output, output_format, silent), num_cpus)


@app.command(help="Handle Input/Output (IO) of TIFF files.")
def tiff(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original TIFF source files."),
    num_cpus: Optional[int] = typer.Option(
        None,
        "-n",
        "--num-cpus",
        help="The number of CPU cores to use for parallel execution.",
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Destination for output file(s)."
    ),
    silent: bool = typer.Option(
        False, "-S", "--silent", help="Disables the progress bars."
    ),
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(write_tiff, expand_sources(paths, output, output_format, silent), num_cpus)


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
