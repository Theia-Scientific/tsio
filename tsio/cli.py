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
from pydantic import BaseModel
from pydicom import dcmread, iter_pixels
from rsciio import digitalmicrograph, emd
from rsciio.image import (
    file_reader as image_file_reader,
    file_writer as image_file_writer,
)
from rsciio.tiff import file_reader as tiff_file_reader
from tifffile import TiffFileError
from tqdm import tqdm
from tsio import __app_name__
from typing import Any, Callable, Dict, List, Optional

LOGGER: logging.Logger = logging.getLogger(__name__)
PREFIX: str = f"{__app_name__.upper()}"

BIT_DEPTH_DTYPE: str = "uint8"
DCM_FILE_EXT: str = ".dcm"
DCM_MIME_TYPE: str = "application/dicom"
DM4_FILE_EXT: str = ".dm4"
DM3_FILE_EXT: str = ".dm3"
DM3_MIME_TYPE: str = "application/vnd.gatan.dm3"
DM4_MIME_TYPE: str = "application/vnd.gatan.dm4"
EMD_FILE_EXT: str = ".emd"
EMD_MIME_TYPE: str = "application/vnd.velox.emd"
JPEG_FILE_EXT: str = ".jpg"
JPEG_MIME_TYPE: str = "image/jpeg"
PNG_FILE_EXT: str = ".png"
PNG_MIME_TYPE: str = "image/png"
TIFF_FILE_EXT: str = ".tif"
TIFF_MIME_TYPE: str = "image/tiff"

mimetypes.add_type(DCM_MIME_TYPE, DCM_FILE_EXT)
mimetypes.add_type(DM3_MIME_TYPE, DM3_FILE_EXT)
mimetypes.add_type(DM4_MIME_TYPE, DM4_FILE_EXT)
mimetypes.add_type(EMD_MIME_TYPE, EMD_FILE_EXT)

logging.getLogger("PIL.Image").setLevel(logging.WARNING)

DELETE_ORIGINAL_OPT: bool = typer.Option(
    False,
    "-D",
    "--delete-original",
    help="Deletes the original file after conversion.",
)


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


class Configuration(BaseModel):
    delete_original: bool = False
    dst: Optional[Path]
    extras: Dict[str, Any] = {}
    output_format: OutputFileFormats
    silent: bool
    src: Path


def map_verbosity(count: int) -> str:
    log_level = "INFO"
    if count >= 1:
        log_level = "DEBUG"
    if count >= 2:
        logging.getLogger("rsciio").setLevel(logging.INFO)
    if count >= 3:
        logging.getLogger("rsciio").setLevel(logging.DEBUG)
    return log_level


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
    delete_original: bool = False,
    normalize: bool = False,
):
    LOGGER.debug(f"{src=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{normalize=}")
    if output is None:
        destination = src.resolve().parent
    else:
        destination = output.resolve()
    pages_count = len(pages)
    LOGGER.debug(f"{pages_count=}")
    src_file_stem = src.stem
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
            max_pixel_intensity = np.max(img)
            LOGGER.debug(f"{max_pixel_intensity=}")
            min_pixel_intensity = np.min(img)
            LOGGER.debug(f"{min_pixel_intensity=}")
            normalization_factor = abs(max_pixel_intensity - min_pixel_intensity)
            LOGGER.debug(f"{normalization_factor=}")
            if normalization_factor > 0:
                img = ((img - min_pixel_intensity) / normalization_factor).astype(
                    np.float32
                )
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
        if delete_original:
            src.unlink(missing_ok=True)


def write_dcm(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    write(
        [
            {
                "data": img,
                "axes": [],
                "index_in_array": None,
                "metadata": {},
                "original_metadata": {},
            }
            for img in iter_pixels(dcmread(cfg.src))
        ],
        cfg.src,
        cfg.dst,
        cfg.output_format,
        cfg.silent,
        normalize=True,
    )


def write_dm(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    try:
        write(
            digitalmicrograph.file_reader(cfg.src),
            cfg.src,
            cfg.dst,
            cfg.output_format,
            cfg.silent,
            normalize=True,
        )
    except NotImplementedError as error:
        LOGGER.warning(f"Skipped '{cfg.src}' because: '{str(error)}'")
    except Exception as error:
        LOGGER.error(f"Skipped '{cfg.src}' because: '{str(error)}'")


def write_emd(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    detector = cfg.extras.get("detector", 0)
    LOGGER.debug(f"{detector=}")
    try:
        emd_data = emd.file_reader(cfg.src, lazy=True, select_type="images")
        LOGGER.debug(f"{emd_data=}")
        LOGGER.debug(f"{len(emd_data)=}")
        if len(emd_data) == 0:
            raise Exception("No image data")
        if "data" not in emd_data[detector]:
            raise Exception("No data field in EMD file")
        dask_data = emd_data[detector]["data"]
        LOGGER.debug(f"{dask_data=}")
        data = dask_data.compute(close_file=True)
        LOGGER.debug(f"{data.shape=}")
        if len(data.shape) == 2:
            pages_count = 1
            pages = [{"data": data, "axes": emd_data[detector]["axes"]}]
        else:
            pages_count = data.shape[0]
            LOGGER.debug(f"{pages_count=}")
            pages = [
                {"data": data[i, ...], "axes": emd_data[detector]["axes"]}
                for i in range(pages_count)
            ]
        write(
            pages,
            cfg.src,
            cfg.dst,
            cfg.output_format,
            cfg.silent,
            normalize=True,
        )
    except Exception as error:
        LOGGER.error(f"Skipped '{cfg.src}' because: '{str(error)}'")


def write_png(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    write(
        image_file_reader(cfg.src),
        cfg.src,
        cfg.dst,
        cfg.output_format,
        cfg.silent,
        normalize=False,
    )


def write_tiff(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    try:
        write(
            tiff_file_reader(cfg.src, multipage_as_list=True),
            cfg.src,
            cfg.dst,
            cfg.output_format,
            cfg.silent,
            normalize=False,
        )
    except TiffFileError:
        if not cfg.silent:
            LOGGER.warning(f"The '{cfg.src}' file is not a TIFF, skipped.")


def expand_sources(
    paths: List[Path],
    output: Optional[Path],
    output_format: OutputFileFormats,
    silent: bool,
    delete_original: bool = False,
    extras: Dict[str, Any] = {},
) -> List[Configuration]:
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{extras=}")
    sources = []
    for path in paths:
        if path.is_dir():
            sources.extend(
                [
                    Configuration(
                        delete_original=delete_original,
                        dst=output,
                        extras=extras,
                        output_format=output_format,
                        silent=silent,
                        src=path.joinpath(p),
                    )
                    for p in os.listdir(path)
                    if path.joinpath(p).is_file()
                ]
            )
        else:
            sources.append(
                Configuration(
                    dst=output,
                    extras=extras,
                    output_format=output_format,
                    silent=silent,
                    src=path,
                )
            )
    return sources


def run(
    write_func: Callable,
    sources: List[Configuration],
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
    delete_original: bool = DELETE_ORIGINAL_OPT,
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
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(
        write_dcm,
        expand_sources(
            paths, output, output_format, silent, delete_original=delete_original
        ),
        num_cpus,
    )


@app.command(help="Handle Input/Output (IO) of DigitalMicrograph (DM) files.")
def dm(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original DM source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
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
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(
        write_dm,
        expand_sources(
            paths, output, output_format, silent, delete_original=delete_original
        ),
        num_cpus,
    )


@app.command(help="Handle Input/Output (IO) of Velox (EMD) files.", name="emd")
def app_emd(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original EMD source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
    detector: int = typer.Option(
        0, "-d", "--detector", help="The index of the detector to export images."
    ),
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
    LOGGER.debug(f"{detector=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{silent=}")
    run(
        write_emd,
        expand_sources(
            paths,
            output,
            output_format,
            silent,
            delete_original=delete_original,
            extras={"detector": detector},
        ),
        num_cpus,
    )


@app.command(help="Handle Input/Output (IO) of PNG files.")
def png(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original PNG source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
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
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(
        write_png,
        expand_sources(
            paths, output, output_format, silent, delete_original=delete_original
        ),
        num_cpus,
    )


@app.command(help="Handle Input/Output (IO) of TIFF files.")
def tiff(
    output_format: OutputFileFormats = typer.Argument(help="The output file format."),
    paths: List[Path] = typer.Argument(help="The original TIFF source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
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
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    run(
        write_tiff,
        expand_sources(
            paths, output, output_format, silent, delete_original=delete_original
        ),
        num_cpus,
    )


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        help="Print debugging statements.",
        envvar=f"{PREFIX}_VERBOSE",
        count=True,
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
    LOGGER.debug(f"{verbose=}")
    LOGGER.debug(f"{version=}")


if __name__ == "__main__":
    app(prog_name=__app_name__)
