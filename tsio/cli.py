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
from pydantic import BaseModel, model_validator, ValidationError
from pydicom import dcmread, iter_pixels
from rich import print
from rsciio import digitalmicrograph, emd
from rsciio.image import (
    file_reader as image_file_reader,
    file_writer as image_file_writer,
)
from rsciio.tiff import file_reader as tiff_file_reader
from rsciio.utils import rgb
from tifffile import TiffFileError
from tqdm import tqdm
from tsio import __app_name__
from typing import Any, Callable, Dict, List, Literal, Optional
from typing_extensions import Self

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

app = typer.Typer(pretty_exceptions_show_locals=False)


class BitDepths(Enum):
    EIGHT = 8
    SIXTEEN = 16

    @property
    def type(self) -> str:
        TYPE_MAP = {BitDepths.EIGHT: "uint8", BitDepths.SIXTEEN: "uint16"}
        return TYPE_MAP[self]

    @property
    def max_pixel_intensity(self) -> int:
        MAX_MAP = {BitDepths.EIGHT: 255, BitDepths.SIXTEEN: 65535}
        return MAX_MAP[self]


class OutputFileFormats(Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

    @property
    def mime_type(self) -> str:
        MIME_TYPES = {
            Self.JPEG: JPEG_MIME_TYPE,
            Self.PNG: PNG_MIME_TYPE,
            Self.TIFF: TIFF_MIME_TYPE,
        }
        return MIME_TYPES[self]

    @property
    def file_ext(self) -> str:
        FILE_EXTS = {
            Self.JPEG: JPEG_FILE_EXT,
            Self.PNG: PNG_FILE_EXT,
            Self.TIFF: TIFF_FILE_EXT,
        }
        return FILE_EXTS[self]

    @property
    def is_alpha_supported(self) -> bool:
        return self != Self.JPEG

    @property
    def is_gray_supported(self) -> bool:
        return self != Self.JPEG


class Output(BaseModel):
    bit_depth: BitDepths
    format: OutputFileFormats
    path: Optional[Path]

    @staticmethod
    def is_gray(img: np.ndarray) -> bool:
        return len(img.shape) == 2

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        max_pixel_intensity = np.max(img)
        LOGGER.debug(f"{max_pixel_intensity=}")
        min_pixel_intensity = np.min(img)
        LOGGER.debug(f"{min_pixel_intensity=}")
        normalization_factor = abs(max_pixel_intensity - min_pixel_intensity)
        LOGGER.debug(f"{normalization_factor=}")
        if normalization_factor > 0:
            return ((img - min_pixel_intensity) / normalization_factor).astype(
                np.float32
            )
        else:
            return img.astype(np.float32)

    def convert(self, img: np.ndarray) -> np.ndarray:
        if Self.is_gray(img) and not self.format.is_gray_supported:
            return cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB,
            )
        elif rgb.is_rgba(img) and not self.format.is_alpha_supported:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            return img

    def cast(self, img: np.ndarray) -> np.ndarray:
        rgbx_or_gray_img = rgb.rgbx2regular_array(img, show_progressbar=False)
        rgbx_or_gray_float_img = Self.normalize(rgbx_or_gray_img)
        rgbx_or_gray_int_img = self.scale(rgbx_or_gray_float_img)
        return Self.convert(rgbx_or_gray_int_img)

    def destination(self, src: Path) -> Path:
        return src.resolve().parent if self.path is None else self.path

    def scale(self, img: np.ndarray) -> np.ndarray:
        return np.round(img * self.bit_depth.max_pixel_intensity).astype(
            self.bit_depth.type
        )

    @model_validator(mode="after")
    def check_supported_bit_depth(self) -> Self:
        SUPPORTED_MAP = {
            OutputFileFormats.JPEG: [BitDepths.EIGHT],
            OutputFileFormats.PNG: [BitDepths.EIGHT, BitDepths.SIXTEEN],
            OutputFileFormats.TIFF: [BitDepths.EIGHT, BitDepths.SIXTEEN],
        }
        if self.bit_depth in SUPPORTED_MAP[self.format]:
            return self
        else:
            raise ValueError(
                (
                    f"The {self.bit_depth.value}-bit depth is not supported "
                    f"for the {self.format.value.upper()} output format."
                )
            )


class Configuration(BaseModel):
    delete_original: bool = False
    extras: Dict[str, Any] = {}
    output: Output
    silent: bool
    src: Path


def print_validation_error(err: ValidationError):
    for e in err.errors():
        msg = e["msg"].removeprefix("Value ")
        print(f"\n[bold red]ERROR![/bold red] {msg}")


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
    output: Output,
    silent: bool,
    delete_original: bool = False,
):
    LOGGER.debug(f"{src=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{silent=}")
    LOGGER.debug(f"{delete_original=}")
    destination = output.destination(src)
    pages_count = len(pages)
    LOGGER.debug(f"{pages_count=}")
    src_file_stem = src.stem
    if pages_count > 1:
        destination = destination.joinpath(src_file_stem)
    os.makedirs(destination, exist_ok=True)
    LOGGER.debug(f"{src_file_stem=}")
    for page_index, page in enumerate(
        tqdm(
            pages,
            total=pages_count,
            desc=src.name,
            disable=silent,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
    ):
        LOGGER.debug(f"{page_index=}")
        if pages_count > 1:
            output_file = destination.joinpath(str(page_index)).with_suffix(
                output.format.file_ext
            )
        else:
            output_file = destination.joinpath(src_file_stem).with_suffix(
                output.format.file_ext
            )
        LOGGER.debug(f"{output_file=}")
        page["data"] = output.cast(page["data"])
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
        cfg.output,
        cfg.silent,
        delete_original=cfg.delete_original,
    )


def write_dm(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    try:
        write(
            digitalmicrograph.file_reader(cfg.src),
            cfg.src,
            cfg.output,
            cfg.silent,
            delete_original=cfg.delete_original,
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
            cfg.output,
            cfg.silent,
            delete_original=cfg.delete_original,
        )
    except Exception as error:
        LOGGER.error(f"Skipped '{cfg.src}' because: '{str(error)}'")


def write_png(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    write(
        image_file_reader(cfg.src),
        cfg.src,
        cfg.output,
        cfg.silent,
        delete_original=cfg.delete_original,
    )


def write_tiff(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    try:
        write(
            tiff_file_reader(cfg.src, multipage_as_list=True),
            cfg.src,
            cfg.output,
            cfg.silent,
            delete_original=cfg.delete_original,
        )
    except TiffFileError:
        if not cfg.silent:
            LOGGER.warning(f"The '{cfg.src}' file is not a TIFF, skipped.")


def expand_sources(
    paths: List[Path],
    output: Output,
    silent: bool,
    delete_original: bool = False,
    extras: Dict[str, Any] = {},
) -> List[Configuration]:
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{output=}")
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
                        extras=extras,
                        output=output,
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
                    delete_original=delete_original,
                    extras=extras,
                    output=output,
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
    LOGGER.debug(f"{sources=}")
    LOGGER.debug(f"{num_cpus=}")
    if platform.system().lower() == "darwin":
        for src in sources:
            write_func(src)
    else:
        with Pool(num_cpus) as pool:
            list(pool.imap(write_func, sources))


DELETE_ORIGINAL_OPT: bool = typer.Option(
    False,
    "-D",
    "--delete-original",
    help="Deletes the original file after conversion.",
)
NUM_CPUS_OPT: Optional[int] = typer.Option(
    None,
    "-n",
    "--num-cpus",
    help="The number of CPU cores to use for parallel execution.",
)
OUTPUT_FORMAT_ARG: OutputFileFormats = typer.Argument(help="The output file format.")
OUTPUT_OPT: Optional[Path] = typer.Option(
    None, "-o", "--output", help="Destination for output file(s)."
)
OUTPUT_BIT_DEPTH_OPT: Literal[8, 16] = typer.Option(
    8,
    "-b",
    "--output-bit-depth",
    help="The bit depth for the output file.",
)
SILENT_OPT: bool = typer.Option(
    False, "-S", "--silent", help="Disables the progress bars."
)


@app.command(help="Handle Input/Output (IO) of DICOM (DCM) files.")
def dcm(
    output_format: OutputFileFormats = OUTPUT_FORMAT_ARG,
    paths: List[Path] = typer.Argument(help="The original DCM source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
    num_cpus: Optional[int] = NUM_CPUS_OPT,
    output: Optional[Path] = OUTPUT_OPT,
    output_bit_depth: int = OUTPUT_BIT_DEPTH_OPT,
    silent: bool = SILENT_OPT,
):
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_bit_depth=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    try:
        run(
            write_dcm,
            expand_sources(
                paths,
                Output(
                    bit_depth=BitDepths(output_bit_depth),
                    format=output_format,
                    path=output,
                ),
                silent,
                delete_original=delete_original,
            ),
            num_cpus,
        )
    except ValidationError as err:
        print_validation_error(err)
        raise typer.Exit(code=1)


@app.command(help="Handle Input/Output (IO) of DigitalMicrograph (DM) files.")
def dm(
    output_format: OutputFileFormats = OUTPUT_FORMAT_ARG,
    paths: List[Path] = typer.Argument(help="The original DM source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
    num_cpus: Optional[int] = NUM_CPUS_OPT,
    output: Optional[Path] = OUTPUT_OPT,
    output_bit_depth: int = OUTPUT_BIT_DEPTH_OPT,
    silent: bool = SILENT_OPT,
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_bit_depth=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    try:
        run(
            write_dm,
            expand_sources(
                paths,
                Output(
                    bit_depth=BitDepths(output_bit_depth),
                    format=output_format,
                    path=output,
                ),
                silent,
                delete_original=delete_original,
            ),
            num_cpus,
        )
    except ValidationError as err:
        print_validation_error(err)
        raise typer.Exit(code=1)


@app.command(help="Handle Input/Output (IO) of Velox (EMD) files.", name="emd")
def app_emd(
    output_format: OutputFileFormats = OUTPUT_FORMAT_ARG,
    paths: List[Path] = typer.Argument(help="The original EMD source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
    detector: int = typer.Option(
        0, "-d", "--detector", help="The index of the detector to export images."
    ),
    num_cpus: Optional[int] = NUM_CPUS_OPT,
    output: Optional[Path] = OUTPUT_OPT,
    output_bit_depth: int = OUTPUT_BIT_DEPTH_OPT,
    silent: bool = SILENT_OPT,
):
    LOGGER.debug(f"{detector=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_bit_depth=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{silent=}")
    try:
        run(
            write_emd,
            expand_sources(
                paths,
                Output(
                    bit_depth=BitDepths(output_bit_depth),
                    format=output_format,
                    path=output,
                ),
                silent,
                delete_original=delete_original,
                extras={"detector": detector},
            ),
            num_cpus,
        )
    except ValidationError as err:
        print_validation_error(err)
        raise typer.Exit(code=1)


@app.command(help="Handle Input/Output (IO) of PNG files.")
def png(
    output_format: OutputFileFormats = OUTPUT_FORMAT_ARG,
    paths: List[Path] = typer.Argument(help="The original PNG source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
    num_cpus: Optional[int] = NUM_CPUS_OPT,
    output: Optional[Path] = OUTPUT_OPT,
    output_bit_depth: int = OUTPUT_BIT_DEPTH_OPT,
    silent: bool = SILENT_OPT,
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_bit_depth=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    try:
        run(
            write_png,
            expand_sources(
                paths,
                Output(
                    bit_depth=BitDepths(output_bit_depth),
                    format=output_format,
                    path=output,
                ),
                silent,
                delete_original=delete_original,
            ),
            num_cpus,
        )
    except ValidationError as err:
        print_validation_error(err)
        raise typer.Exit(code=1)


@app.command(help="Handle Input/Output (IO) of TIFF files.")
def tiff(
    output_format: OutputFileFormats = OUTPUT_FORMAT_ARG,
    paths: List[Path] = typer.Argument(help="The original TIFF source files."),
    delete_original: bool = DELETE_ORIGINAL_OPT,
    num_cpus: Optional[int] = NUM_CPUS_OPT,
    output: Optional[Path] = OUTPUT_OPT,
    output_bit_depth: int = OUTPUT_BIT_DEPTH_OPT,
    silent: bool = SILENT_OPT,
):
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{output_bit_depth=}")
    LOGGER.debug(f"{output_format=}")
    LOGGER.debug(f"{silent=}")
    try:
        run(
            write_tiff,
            expand_sources(
                paths,
                Output(
                    bit_depth=BitDepths(output_bit_depth),
                    format=output_format,
                    path=output,
                ),
                silent,
                delete_original=delete_original,
            ),
            num_cpus,
        )
    except ValidationError as err:
        print_validation_error(err)
        raise typer.Exit(code=1)


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
