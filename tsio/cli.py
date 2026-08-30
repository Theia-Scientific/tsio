#!/usr/bin/env python3

import cv2
import filetype
import importlib.metadata
import logging
import numpy as np
import os
import platform
import typer

from enum import Enum
from filetype.types.image import Dcm, Jpeg, Png, Tiff
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
from tqdm import tqdm
from tsio import __app_name__
from typing import Any, Literal
from typing_extensions import Self

LOGGER: logging.Logger = logging.getLogger(__name__)


class UnsupportedFileType(Exception):
    def __init__(self, src: Path):
        self.src = src


class Dm3(filetype.Type):
    MIME = "application/vnd.gatan.dm3"
    EXTENSION = ".dm3"

    def __init__(self):
        super(Dm3, self).__init__(mime=Dm3.MIME, extension=Dm3.EXTENSION)

    def match(self, buf) -> bool:
        # First 4 bytes are version number = 3
        # Next 4 bytes are the file size
        # Last 4 bytes are "endian"
        return (
            len(buf) > 4
            and buf[0] == 0x00
            and buf[1] == 0x00
            and buf[2] == 0x00
            and buf[3] == 0x03
        )


class Dm4(filetype.Type):
    MIME = "application/vnd.gatan.dm4"
    EXTENSION = ".dm4"

    def __init__(self):
        super(Dm4, self).__init__(mime=Dm4.MIME, extension=Dm4.EXTENSION)

    def match(self, buf) -> bool:
        # First 4 bytes are version number = 4
        # Next 8 bytes are the file size
        # Last 4 bytes are "endian"
        return (
            len(buf) > 4
            and buf[0] == 0x00
            and buf[1] == 0x00
            and buf[2] == 0x00
            and buf[3] == 0x04
        )


class Emd(filetype.Type):
    MIME = "application/vnd.velox.emd"
    EXTENSION = ".emd"

    def __init__(self):
        super(Emd, self).__init__(mime=Emd.MIME, extension=Emd.EXTENSION)

    def match(self, buf) -> bool:
        # Velox EMD is a HDF5 file.
        return (
            len(buf) > 7
            and buf[0] == 0x89
            and buf[1] == 0x48
            and buf[2] == 0x44
            and buf[3] == 0x46
            and buf[4] == 0x0D
            and buf[5] == 0x0A
            and buf[6] == 0x1A
            and buf[7] == 0x0A
        )


filetype.add_type(Dm3())
filetype.add_type(Dm4())
filetype.add_type(Emd())


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


class ToFormats(Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

    @property
    def mime_type(self) -> str:
        MIME_TYPES = {
            ToFormats.JPEG: Jpeg.MIME,
            ToFormats.PNG: Png.MIME,
            ToFormats.TIFF: Tiff.MIME,
        }
        return MIME_TYPES[self]

    @property
    def file_ext(self) -> str:
        FILE_EXTS = {
            ToFormats.JPEG: "." + Jpeg.EXTENSION,
            ToFormats.PNG: "." + Png.EXTENSION,
            ToFormats.TIFF: "." + Tiff.EXTENSION,
        }
        return FILE_EXTS[self]

    @property
    def is_alpha_supported(self) -> bool:
        return self != ToFormats.JPEG

    @property
    def is_gray_supported(self) -> bool:
        return self != ToFormats.JPEG


class Output(BaseModel):
    bit_depth: BitDepths
    path: Path | None
    format: ToFormats

    @staticmethod
    def is_gray(img: np.ndarray) -> bool:
        return len(img.shape) == 2

    @staticmethod
    def is_rgb(img: np.ndarray) -> bool:
        return len(img.shape) == 3

    @staticmethod
    def is_rgba(img: np.ndarray) -> bool:
        return Output.is_rgb(img) and img.shape[2] == 4

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        max_pixel_intensity = int(np.max(img))
        LOGGER.debug(f"{max_pixel_intensity=}")
        min_pixel_intensity = int(np.min(img))
        if max_pixel_intensity == min_pixel_intensity:
            min_pixel_intensity = 0
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
        if Output.is_gray(img) and not self.format.is_gray_supported:
            return cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB,
            )
        elif Output.is_rgba(img) and not self.format.is_alpha_supported:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            return img

    def cast(self, img: np.ndarray) -> np.ndarray:
        rgbx_or_gray_img = rgb.rgbx2regular_array(img, show_progressbar=False)
        rgbx_or_gray_float_img = Output.normalize(rgbx_or_gray_img)
        rgbx_or_gray_int_img = self.scale(rgbx_or_gray_float_img)
        return self.convert(rgbx_or_gray_int_img)

    def destination(self, src: Path) -> Path:
        return src.resolve().parent if self.path is None else self.path

    def scale(self, img: np.ndarray) -> np.ndarray:
        return np.round(img * self.bit_depth.max_pixel_intensity).astype(
            self.bit_depth.type
        )

    @model_validator(mode="after")
    def check_supported_bit_depth(self) -> Self:
        SUPPORTED_MAP = {
            ToFormats.JPEG: [BitDepths.EIGHT],
            ToFormats.PNG: [BitDepths.EIGHT, BitDepths.SIXTEEN],
            ToFormats.TIFF: [BitDepths.EIGHT, BitDepths.SIXTEEN],
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
    delete_original: bool
    extras: dict[str, Any] | None
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
    pages: list[dict[str, Any]],
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
            disable=silent or pages_count == 1,
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


def run_dcm(cfg: Configuration):
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


def run_dm(cfg: Configuration):
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


def run_emd(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    if cfg.extras is None:
        detector = 0
    else:
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


def run_png(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    write(
        image_file_reader(cfg.src),
        cfg.src,
        cfg.output,
        cfg.silent,
        delete_original=cfg.delete_original,
    )


def run_tiff(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    write(
        tiff_file_reader(cfg.src, multipage_as_list=True),
        cfg.src,
        cfg.output,
        cfg.silent,
        delete_original=cfg.delete_original,
    )


def expand_sources(
    paths: list[Path],
    output: Output,
    silent: bool,
    delete_original: bool = False,
    extras: dict[str, Any] | None = None,
) -> list[Configuration]:
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


def run(cfg: Configuration):
    LOGGER.debug(f"{cfg=}")
    kind = filetype.guess(cfg.src)
    LOGGER.debug(f"{kind=}")
    if kind is None:
        raise UnsupportedFileType(cfg.src)
    else:
        SUPPORTED_MAP = {
            Dcm.MIME: run_dcm,
            Dm3.MIME: run_dm,
            Dm4.MIME: run_dm,
            Emd.MIME: run_emd,
            Png.MIME: run_png,
            Tiff.MIME: run_tiff,
        }
        runner = SUPPORTED_MAP.get(kind.mime)
        if runner is None:
            raise UnsupportedFileType(cfg.src)
        else:
            runner(cfg)


PROGRESS_BAR_FORMAT: str = "{l_bar}{bar}| {n_fmt}/{total_fmt}"

DELETE_ORIGINAL_OPT: bool = typer.Option(
    False,
    "-D",
    "--delete-original",
    help="Deletes the original file after conversion.",
)
NUM_CPUS_OPT: int | None = typer.Option(
    None,
    "-n",
    "--num-cpus",
    help="The number of CPU cores to use for parallel execution.",
)
OUTPUT_OPT: Path | None = typer.Option(
    None, "-o", "--output", help="Destination for output file(s)."
)
PATHS_ARG: list[Path] = typer.Argument(help="The original source files.")
SILENT_OPT: bool = typer.Option(
    False, "-S", "--silent", help="Disables the progress bars."
)
TO_BIT_DEPTH_OPT: Literal[8, 16] = typer.Option(
    8,
    "-b",
    "--to-bit-depth",
    help="The bit depth for the output file.",
)
TO_FORMAT_OPT: ToFormats = typer.Option(
    ToFormats.JPEG.value,
    "-t",
    "--to",
    case_sensitive=False,
    help="The output file format.",
)
VERBOSE_OPT: int = typer.Option(
    0,
    "--verbose",
    "-v",
    help="Print debugging statements.",
    count=True,
)
VERSION_OPT: bool | None = typer.Option(
    None,
    "--version",
    help="Prints the version.",
    callback=version_callback,
    is_eager=True,
)


@app.command()
def main(
    paths: list[Path] = PATHS_ARG,
    delete_original: bool = DELETE_ORIGINAL_OPT,
    num_cpus: int | None = NUM_CPUS_OPT,
    output: Path | None = OUTPUT_OPT,
    silent: bool = SILENT_OPT,
    to_bit_depth: int = TO_BIT_DEPTH_OPT,
    to_format: ToFormats = TO_FORMAT_OPT,
    verbose: int = VERBOSE_OPT,
    version: bool | None = VERSION_OPT,
):
    logging.basicConfig(level=map_verbosity(verbose))
    LOGGER.debug(f"{delete_original=}")
    LOGGER.debug(f"{paths=}")
    LOGGER.debug(f"{num_cpus=}")
    LOGGER.debug(f"{output=}")
    LOGGER.debug(f"{silent=}")
    LOGGER.debug(f"{to_bit_depth=}")
    LOGGER.debug(f"{to_format=}")
    LOGGER.debug(f"{verbose=}")
    LOGGER.debug(f"{version=}")
    try:
        sources = expand_sources(
            paths,
            Output(
                bit_depth=BitDepths(to_bit_depth),
                format=to_format,
                path=output,
            ),
            silent,
            delete_original=delete_original,
        )
        if platform.system().lower() == "darwin":
            _ = list(
                tqdm(
                    map(run, sources),
                    bar_format=PROGRESS_BAR_FORMAT,
                    disable=silent,
                    total=len(sources),
                )
            )
        else:
            with Pool(num_cpus) as pool:
                _ = list(
                    tqdm(
                        pool.imap(run, sources),
                        bar_format=PROGRESS_BAR_FORMAT,
                        disable=silent,
                        total=len(sources),
                    )
                )
    except UnsupportedFileType as err:
        LOGGER.warning(f"The {err.src} file is not supported. Skipping!")
    except ValidationError as err:
        print_validation_error(err)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app(prog_name=__app_name__)
