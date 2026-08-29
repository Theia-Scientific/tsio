#!/usr/bin/env python3

import cv2
import datetime
import filetype
import gdown
import importlib.metadata
import numpy as np
import os
import pytest

from pathlib import Path
from pydantic import ValidationError
from pydicom import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian
from pytest_mock import MockerFixture
from rsciio.digitalmicrograph import file_reader as dm_file_reader
from rsciio.image import file_writer as image_file_writer
from rsciio.tiff import file_reader as tiff_file_reader, file_writer as tiff_file_writer
from rsciio.utils import rgb
from tsio import __app_name__
from tsio.cli import (
    app,
    BitDepths,
    Configuration,
    DCM_FILE_EXT,
    DM3_FILE_EXT,
    DM4_FILE_EXT,
    EMD_FILE_EXT,
    expand_sources,
    FromFormats,
    map_verbosity,
    JPEG_FILE_EXT,
    JPEG_MIME_TYPE,
    Output,
    PNG_FILE_EXT,
    PNG_MIME_TYPE,
    run,
    run_dcm,
    run_dm,
    run_emd,
    run_png,
    run_tiff,
    TIFF_FILE_EXT,
    TIFF_MIME_TYPE,
    ToFormats,
    write,
)
from typer.testing import CliRunner
from typing import Any, Callable

runner = CliRunner()


@pytest.fixture
def black_8bit_gray_image() -> np.ndarray:
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def black_8bit_rgb_image() -> np.ndarray:
    return np.zeros((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def black_8bit_rgba_image() -> np.ndarray:
    return np.zeros((256, 256, 4), dtype=np.uint8)


@pytest.fixture
def black_16bit_gray_image() -> np.ndarray:
    return np.zeros((256, 256), dtype=np.uint16)


@pytest.fixture
def black_16bit_rgb_image() -> np.ndarray:
    return np.zeros((256, 256, 3), dtype=np.uint16)


@pytest.fixture
def black_16bit_rgba_image() -> np.ndarray:
    return np.zeros((256, 256, 4), dtype=np.uint16)


@pytest.fixture
def black_8bit_gray_png(black_8bit_gray_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": black_8bit_gray_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256)
    return png_file


@pytest.fixture
def black_8bit_rgb_png(black_8bit_rgb_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": black_8bit_rgb_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 3)
    return png_file


@pytest.fixture
def black_8bit_rgba_png(black_8bit_rgba_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": black_8bit_rgba_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 4)
    return png_file


@pytest.fixture
def black_16bit_gray_png(black_16bit_gray_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": black_16bit_gray_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint16"
    assert png_img.shape == (256, 256)
    return png_file


@pytest.fixture
def black_16bit_rgb_png(black_16bit_rgb_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    # Pillow does not support RGB 16-bit, but the PNG specification does
    # support it. Pillow is used by rosettasciio's `image_file_writer`.
    write_result = cv2.imwrite(
        str(png_file), cv2.cvtColor(black_16bit_rgb_image, cv2.COLOR_RGB2BGR)
    )
    assert write_result
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint16"
    assert png_img.shape == (256, 256, 3)
    return png_file


@pytest.fixture
def black_16bit_rgba_png(black_16bit_rgba_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    write_result = cv2.imwrite(
        str(png_file), cv2.cvtColor(black_16bit_rgba_image, cv2.COLOR_RGBA2BGRA)
    )
    assert write_result
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint16"
    assert png_img.shape == (256, 256, 4)
    return png_file


@pytest.fixture
def white_8bit_gray_image() -> np.ndarray:
    return np.ones((256, 256), dtype=np.uint8)


@pytest.fixture
def white_8bit_rgb_image() -> np.ndarray:
    return np.ones((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def white_8bit_rgba_image() -> np.ndarray:
    return np.ones((256, 256, 4), dtype=np.uint8)


@pytest.fixture
def white_8bit_gray_png(white_8bit_gray_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": white_8bit_gray_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 4)
    return png_file


@pytest.fixture
def white_8bit_rgb_png(white_8bit_rgb_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": white_8bit_rgb_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 3)
    return png_file


@pytest.fixture
def white_8bit_rgba_png(white_8bit_rgba_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": white_8bit_rgba_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 4)
    return png_file


@pytest.fixture
def white_16bit_gray_image() -> np.ndarray:
    return np.ones((256, 256), dtype=np.uint16)


@pytest.fixture
def white_16bit_rgb_image() -> np.ndarray:
    return np.ones((256, 256, 3), dtype=np.uint16)


@pytest.fixture
def white_16bit_rgba_image() -> np.ndarray:
    return np.ones((256, 256, 4), dtype=np.uint16)


@pytest.fixture
def white_16bit_gray_png(white_16bit_gray_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": white_16bit_gray_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 4)
    return png_file


@pytest.fixture
def white_16bit_rgb_png(white_16bit_rgb_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": white_16bit_rgb_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 3)
    return png_file


@pytest.fixture
def white_16bit_rgba_png(white_16bit_rgba_image: np.ndarray, tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": white_16bit_rgba_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    assert png_file.exists()
    png_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256, 4)
    return png_file


@pytest.fixture
def random_16bit_multipage_image() -> np.ndarray:
    return np.random.randint(0, 2**12, (64, 301, 219), "uint16")


@pytest.fixture
def black_8bit_gray_single_page_tiff(
    black_8bit_gray_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {"data": black_8bit_gray_image}
    tiff_file_writer(str(tif_file), signal)
    assert tif_file.exists()
    tif_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    assert tif_img is not None
    assert tif_img.dtype.name == "uint8"
    assert tif_img.shape == (256, 256)
    return tif_file


@pytest.fixture
def black_8bit_rgb_single_page_tiff(
    black_8bit_rgb_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {"data": rgb.regular_array2rgbx(black_8bit_rgb_image)}
    tiff_file_writer(str(tif_file), signal)
    assert tif_file.exists()
    tif_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    assert tif_img is not None
    assert tif_img.dtype.name == "uint8"
    assert tif_img.shape == (256, 256, 3)
    return tif_file


@pytest.fixture
def black_8bit_rgba_single_page_tiff(
    black_8bit_rgba_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {"data": rgb.regular_array2rgbx(black_8bit_rgba_image)}
    tiff_file_writer(str(tif_file), signal)
    assert tif_file.exists()
    tif_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    assert tif_img is not None
    assert tif_img.dtype.name == "uint8"
    assert tif_img.shape == (256, 256, 4)
    return tif_file


@pytest.fixture
def black_16bit_gray_single_page_tiff(
    black_16bit_gray_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {"data": black_16bit_gray_image}
    tiff_file_writer(str(tif_file), signal)
    assert tif_file.exists()
    tif_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    assert tif_img is not None
    assert tif_img.dtype.name == "uint16"
    assert tif_img.shape == (256, 256)
    return tif_file


@pytest.fixture
def black_16bit_rgb_single_page_tiff(
    black_16bit_rgb_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {"data": rgb.regular_array2rgbx(black_16bit_rgb_image)}
    tiff_file_writer(str(tif_file), signal)
    assert tif_file.exists()
    tif_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    assert tif_img is not None
    assert tif_img.dtype.name == "uint16"
    assert tif_img.shape == (256, 256, 3)
    return tif_file


@pytest.fixture
def black_16bit_rgba_single_page_tiff(
    black_16bit_rgba_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {"data": rgb.regular_array2rgbx(black_16bit_rgba_image)}
    tiff_file_writer(str(tif_file), signal)
    assert tif_file.exists()
    tif_img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    assert tif_img is not None
    assert tif_img.dtype.name == "uint16"
    assert tif_img.shape == (256, 256, 3)
    return tif_file


@pytest.fixture
def random_multipage_tiff(
    random_16bit_multipage_image: np.ndarray, tmp_path: Path
) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {
        "data": random_16bit_multipage_image,
    }
    tiff_file_writer(str(tif_file), signal)
    return tif_file


@pytest.fixture
def tmp_assets() -> Path:
    cwd = Path(os.getcwd())
    assets = cwd.joinpath(".tmp", "tests", "assets")
    if not assets.exists():
        os.makedirs(assets, exist_ok=True)
    return assets


@pytest.fixture
def assets() -> Path:
    return Path(os.getcwd()).joinpath("tests", "assets")


@pytest.fixture
def heart_png(assets: Path) -> Path:
    return assets.joinpath("heart16.png")


@pytest.fixture
def ncsu_tif(assets: Path) -> Path:
    return assets.joinpath("1138622_small_slice_42.tif")


@pytest.fixture
def v09_loaded_confusion_matrix_png(assets: Path) -> Path:
    return assets.joinpath("v09_loaded_confusion_matrix.png")


@pytest.fixture
def dcm(black_8bit_gray_image: np.ndarray, tmp_path: Path) -> Path:
    height, width = black_8bit_gray_image.shape
    grey_img = black_8bit_gray_image
    dcm_file = tmp_path.joinpath("test.dcm")
    ds = Dataset()
    ds.Rows = height
    ds.Columns = width
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.SamplesPerPixel = 1
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = grey_img.tobytes()
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S.%f")
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.ImplementationClassUID = UID("1.2.3.4")
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta
    ds.save_as(dcm_file, enforce_file_format=True)
    return dcm_file


@pytest.fixture
def dm3(tmp_assets: Path) -> Path:
    dst = tmp_assets.joinpath("2.dm3")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1BDNBta7cSMUtmb4r5e5gvqgBBhRW6xRq"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def dm4(tmp_assets: Path) -> Path:
    dst = tmp_assets.joinpath("1.dm4")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1Pkbfnl5-7zVSB1h7JfwLbKy6yOxxMvR-"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def single_image_emd(tmp_assets: Path) -> Path:
    dst = tmp_assets.joinpath("3.emd")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1Z-aJUxQdpzd4v5ptOZvIY5Q8EYYSGi7L"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def multiple_images_emd(tmp_assets: Path) -> Path:
    dst = tmp_assets.joinpath("4.emd")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1GDB9hvN1FAULy1JwQN0THL7fAs4BbTlf"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def run_cfg(
    output_cfg: Callable[..., Output],
) -> Callable[
    [Path, bool, dict[str, Any], FromFormats | None, Output, bool], Configuration
]:
    DEFAULT_OUTPUT = output_cfg()

    def _make_run_cfg(
        src: Path,
        delete_original: bool = False,
        extras: dict[str, Any] | None = None,
        from_format: FromFormats | None = None,
        output: Output = DEFAULT_OUTPUT,
        silent: bool = True,
    ) -> Configuration:
        return Configuration(
            delete_original=delete_original,
            extras=extras,
            from_format=from_format,
            output=output,
            silent=silent,
            src=src,
        )

    return _make_run_cfg


@pytest.fixture
def output_cfg() -> Callable[[BitDepths, ToFormats, Path], Output]:
    def _make_output_cfg(
        bit_depth: BitDepths = BitDepths.EIGHT,
        format: ToFormats = ToFormats.JPEG,
        path: Path | None = None,
    ) -> Output:
        return Output(bit_depth=bit_depth, format=format, path=path)

    return _make_output_cfg


def test_bit_depths_type():
    assert BitDepths.EIGHT.type == "uint8"
    assert BitDepths.SIXTEEN.type == "uint16"


def test_bit_depths_max_pixel_intensity():
    assert BitDepths.EIGHT.max_pixel_intensity == 255
    assert BitDepths.SIXTEEN.max_pixel_intensity == 65535


def test_from_formats_ext():
    assert FromFormats.DCM.ext == DCM_FILE_EXT
    assert FromFormats.DM3.ext == DM3_FILE_EXT
    assert FromFormats.DM4.ext == DM4_FILE_EXT
    assert FromFormats.EMD.ext == EMD_FILE_EXT
    assert FromFormats.PNG.ext == PNG_FILE_EXT
    assert FromFormats.TIFF.ext == TIFF_FILE_EXT


def test_to_formats_is_alpha_supported():
    assert not ToFormats.JPEG.is_alpha_supported
    assert ToFormats.PNG.is_alpha_supported
    assert ToFormats.TIFF.is_alpha_supported


def test_to_formats_is_gray_supported():
    assert not ToFormats.JPEG.is_gray_supported
    assert ToFormats.PNG.is_gray_supported
    assert ToFormats.TIFF.is_gray_supported


def test_to_formats_mime_type():
    assert ToFormats.JPEG.mime_type == JPEG_MIME_TYPE
    assert ToFormats.PNG.mime_type == PNG_MIME_TYPE
    assert ToFormats.TIFF.mime_type == TIFF_MIME_TYPE


def test_to_formats_file_ext():
    assert ToFormats.JPEG.file_ext == JPEG_FILE_EXT
    assert ToFormats.PNG.file_ext == PNG_FILE_EXT
    assert ToFormats.TIFF.file_ext == TIFF_FILE_EXT


def test_output_destination_with_none_path(
    tmp_path: Path, output_cfg: Callable[..., Output]
):
    assert output_cfg().destination(tmp_path) == tmp_path.resolve().parent


def test_output_destination_with_path(
    tmp_path: Path, output_cfg: Callable[..., Output]
):
    assert output_cfg(path=tmp_path).destination(tmp_path) == tmp_path


def test_output_is_gray(
    black_8bit_gray_image: np.ndarray,
    black_16bit_gray_image: np.ndarray,
    black_8bit_rgb_image: np.ndarray,
    black_16bit_rgb_image: np.ndarray,
    black_8bit_rgba_image: np.ndarray,
    black_16bit_rgba_image: np.ndarray,
):
    assert Output.is_gray(black_8bit_gray_image)
    assert Output.is_gray(black_16bit_gray_image)
    assert not Output.is_gray(black_8bit_rgb_image)
    assert not Output.is_gray(black_16bit_rgb_image)
    assert not Output.is_gray(black_8bit_rgba_image)
    assert not Output.is_gray(black_16bit_rgba_image)


def test_output_cast_eight_gray(
    black_16bit_gray_image: np.ndarray,
    output_cfg: Callable[..., Output],
    tmp_path: Path,
):
    img = output_cfg(path=tmp_path).cast(black_16bit_gray_image)
    assert img.dtype.name == "uint8"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_eight_rgb(
    black_16bit_rgb_image: np.ndarray, output_cfg: Callable[..., Output], tmp_path: Path
):
    img = output_cfg(path=tmp_path).cast(black_16bit_rgb_image)
    assert img.dtype.name == "uint8"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_eight_rgba(
    black_16bit_rgba_image: np.ndarray,
    output_cfg: Callable[..., Output],
    tmp_path: Path,
):
    img = output_cfg(path=tmp_path).cast(black_16bit_rgba_image)
    assert img.dtype.name == "uint8"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_sixteen_gray(
    black_16bit_gray_image: np.ndarray,
    output_cfg: Callable[..., Output],
    tmp_path: Path,
):
    img = output_cfg(
        bit_depth=BitDepths.SIXTEEN, format=ToFormats.PNG, path=tmp_path
    ).cast(black_16bit_gray_image)
    assert img.dtype.name == "uint16"
    assert img.shape == (256, 256)
    assert not np.any(img)


def test_output_cast_sixteen_rgb(
    black_16bit_rgb_image: np.ndarray, output_cfg: Callable[..., Output], tmp_path: Path
):
    img = output_cfg(
        bit_depth=BitDepths.SIXTEEN, format=ToFormats.PNG, path=tmp_path
    ).cast(black_16bit_rgb_image)
    assert img.dtype.name == "uint16"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_sixteen_rgba(
    black_16bit_rgba_image: np.ndarray,
    output_cfg: Callable[..., Output],
    tmp_path: Path,
):
    img = output_cfg(
        bit_depth=BitDepths.SIXTEEN, format=ToFormats.PNG, path=tmp_path
    ).cast(black_16bit_rgba_image)
    assert img.dtype.name == "uint16"
    assert img.shape == (256, 256, 4)
    assert not np.any(img)


def test_output_validation():
    with pytest.raises(ValidationError):
        _ = Output(bit_depth=BitDepths.SIXTEEN, format=ToFormats.JPEG, path=None)


def test_map_verbosity_none():
    actual = map_verbosity(0)
    assert actual == "INFO"


def test_map_verbosity_one():
    actual = map_verbosity(1)
    assert actual == "DEBUG"


def test_map_verbosity_two():
    import logging

    actual = map_verbosity(2)
    assert logging.getLogger("rsciio").level == logging.INFO
    assert actual == "DEBUG"


def test_map_verbosity_three():
    import logging

    actual = map_verbosity(3)
    assert logging.getLogger("rsciio").level == logging.DEBUG
    assert actual == "DEBUG"


def test_write_tiff_with_black_8bit_gray(
    black_8bit_gray_single_page_tiff: Path, output_cfg: Callable[..., Output]
):
    src = black_8bit_gray_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output_cfg(),
        True,
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_write_tiff_with_black_8bit_rgb(
    black_8bit_rgb_single_page_tiff: Path, output_cfg: Callable[..., Output]
):
    src = black_8bit_rgb_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output_cfg(),
        True,
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_write_tiff_with_black_8bit_rgba(
    black_8bit_rgba_single_page_tiff: Path, output_cfg: Callable[..., Output]
):
    src = black_8bit_rgba_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output_cfg(),
        True,
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_write_tiff_with_black_16bit_gray(
    black_16bit_gray_single_page_tiff: Path, output_cfg: Callable[..., Output]
):
    src = black_16bit_gray_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output_cfg(),
        True,
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_write_with_single_tiff_delete_original(
    black_16bit_gray_single_page_tiff: Path, output_cfg: Callable[..., Output]
):
    src = black_16bit_gray_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output_cfg(),
        True,
        delete_original=True,
    )
    assert dst.exists()
    assert not src.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_write_with_multipages_tiff(
    output_cfg: Callable[..., Output],
    random_multipage_tiff: Path,
    random_16bit_multipage_image: np.ndarray,
):
    pages_count, *_ = random_16bit_multipage_image.shape
    src = random_multipage_tiff
    src_stem = src.stem
    dst = src.parent.joinpath(src_stem)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        output_cfg(),
        True,
    )
    assert dst.exists()
    assert (
        len([name for name in os.listdir(dst) if dst.joinpath(name).is_file()])
        == pages_count
    )
    for name in os.listdir(dst):
        assert dst.joinpath(name).exists()
        kind = filetype.guess(str(dst.joinpath(name)))
        assert kind is not None
        assert kind.mime == "image/jpeg"
        jpeg_img = cv2.imread(str(dst.joinpath(name)), cv2.IMREAD_UNCHANGED)
        assert jpeg_img is not None
        assert jpeg_img.dtype.name == "uint8"
        assert len(jpeg_img.shape) == 3
        assert np.any(jpeg_img)


def test_write_with_dm3(dm3: Path, output_cfg: Callable[..., Output], tmp_path: Path):
    src = dm3
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    write(
        dm_file_reader(src),
        src,
        output_cfg(path=tmp_path),
        True,
    )
    assert dst.exists()


def test_write_with_dm4(dm4: Path, output_cfg: Callable[..., Output], tmp_path: Path):
    src = dm4
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    write(
        dm_file_reader(src),
        src,
        output_cfg(path=tmp_path),
        True,
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1024, 1024, 3)
    assert np.any(jpeg_img)


def test_run_dcm(
    dcm: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = dcm
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    run_dcm(run_cfg(src, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_dm(
    dm4: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = dm4
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    run_dm(run_cfg(src, output=output_cfg(path=dst)))
    assert dst.exists()


def test_run_dm_fails_with_not_implemented(
    mocker: MockerFixture, run_cfg: Callable[..., Configuration], tmp_path: Path
):
    src = tmp_path.joinpath("test.dm4")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args: Any, **kwargs: Any):
        _ = args
        _ = kwargs

        raise NotImplementedError("Not supported version")

    _ = mocker.patch("rsciio.digitalmicrograph.file_reader", mock_file_reader)
    run_dm(run_cfg(src))
    assert not dst.exists()


def test_run_dm_fails_with_exception(
    run_cfg: Callable[..., Configuration], tmp_path: Path
):
    src = tmp_path.joinpath("test.dm4")
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_dm(run_cfg(src))
    assert not dst.exists()


def test_run_emd_single_image(
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    single_image_emd: Path,
    tmp_path: Path,
):
    src = single_image_emd
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    run_emd(run_cfg(src, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1024, 1024, 3)
    assert np.any(jpeg_img)


def test_run_emd_multiple_images(
    multiple_images_emd: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = multiple_images_emd
    dst = tmp_path.joinpath(src.stem)
    run_emd(run_cfg(src, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    assert os.path.isdir(dst)
    assert len(os.listdir(dst)) == 37
    for name in os.listdir(dst):
        dst_file = dst.joinpath(name)
        assert dst_file.exists()
        kind = filetype.guess(str(dst_file))
        assert kind is not None
        assert kind.mime == "image/jpeg"
        jpeg_img = cv2.imread(str(dst_file), cv2.IMREAD_UNCHANGED)
        assert jpeg_img is not None
        assert jpeg_img.dtype.name == "uint8"
        assert len(jpeg_img.shape) == 3


def test_run_emd_single_image_with_extras(
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    single_image_emd: Path,
    tmp_path: Path,
):
    src = single_image_emd
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    run_emd(run_cfg(src, extras={"detector": 0}, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1024, 1024, 3)
    assert np.any(jpeg_img)


def test_run_emd_with_no_image_data(
    mocker: MockerFixture, run_cfg: Callable[..., Configuration], tmp_path: Path
):
    src = tmp_path.joinpath("test.emd")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args: Any, **kwargs: Any) -> list[Any]:
        _ = args
        _ = kwargs

        return []

    _ = mocker.patch("rsciio.emd.file_reader", mock_file_reader)
    run_emd(run_cfg(src))
    assert not dst.exists()


def test_run_emd_with_no_data_field(
    mocker: MockerFixture, run_cfg: Callable[..., Configuration], tmp_path: Path
):
    src = tmp_path.joinpath("test.emd")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args: Any, **kwargs: Any) -> list[dict[str, list[Any]]]:
        _ = args
        _ = kwargs

        return [{"axes": []}]

    _ = mocker.patch("rsciio.emd.file_reader", mock_file_reader)
    run_emd(run_cfg(src))
    assert not dst.exists()


def test_run_emd_fails_with_exception(
    mocker: MockerFixture, run_cfg: Callable[..., Configuration], tmp_path: Path
):
    src = tmp_path.joinpath("test.emd")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args: Any, **kwargs: Any):
        _ = args
        _ = kwargs

        raise Exception("Test Exception")

    _ = mocker.patch("rsciio.emd.file_reader", mock_file_reader)
    run_emd(run_cfg(src))
    assert not dst.exists()


def test_run_with_from_format_value(
    black_8bit_gray_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = black_8bit_gray_png
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    run(run_cfg(src, from_format=FromFormats.PNG, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_with_from_format_none(
    black_8bit_gray_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = black_8bit_gray_png
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    run(run_cfg(src, from_format=None, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_png_8bit_gray(
    black_8bit_gray_png: Path, run_cfg: Callable[..., Configuration]
):
    src = black_8bit_gray_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_png_8bit_rgb(
    black_8bit_rgb_png: Path, run_cfg: Callable[..., Configuration]
):
    src = black_8bit_rgb_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_png_8bit_rgba(
    black_8bit_rgba_png: Path, run_cfg: Callable[..., Configuration]
):
    src = black_8bit_rgba_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_png_16bit_gray(
    black_16bit_gray_png: Path, run_cfg: Callable[..., Configuration]
):
    src = black_16bit_gray_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_png_16bit_rgb(
    black_16bit_rgb_png: Path, run_cfg: Callable[..., Configuration]
):
    src = black_16bit_rgb_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_png_16bit_rgba(
    black_16bit_rgba_png: Path, run_cfg: Callable[..., Configuration]
):
    src = black_16bit_rgba_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_8bit_grayscale_png_to_8bit_tiff(
    black_8bit_gray_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
):
    src = black_8bit_gray_png
    dst = src.with_suffix(TIFF_FILE_EXT)
    run_png(run_cfg(src, output=output_cfg(format=ToFormats.TIFF)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/tiff"
    tiff_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert tiff_img is not None
    assert tiff_img.dtype.name == "uint8"
    assert tiff_img.shape == (256, 256)
    assert not np.any(tiff_img)


def test_run_8bit_grayscale_png_to_16bit_tiff(
    black_8bit_gray_image: np.ndarray,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = tmp_path.joinpath("tmp.png")
    write_result = cv2.imwrite(str(src), black_8bit_gray_image)
    assert write_result
    assert src.exists()
    png_img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    assert png_img is not None
    assert png_img.dtype.name == "uint8"
    assert png_img.shape == (256, 256)
    dst = src.with_suffix(TIFF_FILE_EXT)
    run_png(
        run_cfg(
            src, output=output_cfg(bit_depth=BitDepths.SIXTEEN, format=ToFormats.TIFF)
        )
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/tiff"
    tiff_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert tiff_img is not None
    assert tiff_img.dtype.name == "uint16"
    assert tiff_img.shape == (256, 256)
    assert not np.any(tiff_img)


def test_run_16bit_grayscale_png_to_8bit_tiff(
    black_16bit_gray_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
):
    src = black_16bit_gray_png
    dst = src.with_suffix(TIFF_FILE_EXT)
    run_png(run_cfg(src, output=output_cfg(format=ToFormats.TIFF)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/tiff"
    tiff_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert tiff_img is not None
    assert tiff_img.dtype.name == "uint8"
    assert tiff_img.shape == (256, 256)
    assert not np.any(tiff_img)


def test_run_16bit_grayscale_png_to_16bit_tiff(
    black_16bit_gray_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
):
    src = black_16bit_gray_png
    dst = src.with_suffix(TIFF_FILE_EXT)
    run_png(
        run_cfg(
            src, output=output_cfg(bit_depth=BitDepths.SIXTEEN, format=ToFormats.TIFF)
        )
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/tiff"
    tiff_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert tiff_img is not None
    assert tiff_img.dtype.name == "uint16"
    assert tiff_img.shape == (256, 256)
    assert not np.any(tiff_img)


def test_run_heart_png_to_8bit_gray_tiff(
    heart_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = heart_png
    dst = tmp_path.joinpath(src.name).with_suffix(TIFF_FILE_EXT)
    run_png(
        run_cfg(
            src,
            output=output_cfg(
                bit_depth=BitDepths.EIGHT, format=ToFormats.TIFF, path=tmp_path
            ),
        )
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/tiff"
    tiff_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert tiff_img is not None
    assert tiff_img.dtype.name == "uint8"
    assert tiff_img.shape == (256, 256)
    assert np.any(tiff_img)


def test_run_heart_png_to_16bit_gray_tiff(
    heart_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = heart_png
    dst = tmp_path.joinpath(src.name).with_suffix(TIFF_FILE_EXT)
    run_png(
        run_cfg(
            src,
            output=output_cfg(
                bit_depth=BitDepths.SIXTEEN,
                format=ToFormats.TIFF,
                path=tmp_path,
            ),
        )
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/tiff"
    tiff_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert tiff_img is not None
    assert tiff_img.dtype.name == "uint16"
    assert tiff_img.shape == (256, 256)
    assert np.any(tiff_img)


def test_run_confusion_matrix_png_to_jpeg(
    v09_loaded_confusion_matrix_png: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    src = v09_loaded_confusion_matrix_png
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src, output=output_cfg(path=tmp_path)))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1606, 1840, 3)
    assert np.any(jpeg_img)


def test_run_white_8bit_rgba_png(
    white_8bit_rgba_png: Path, run_cfg: Callable[..., Configuration]
):
    src = white_8bit_rgba_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_png(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert np.all(jpeg_img == 255)


def test_run_tiff(
    black_16bit_gray_single_page_tiff: Path, run_cfg: Callable[..., Configuration]
):
    src = black_16bit_gray_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    run_tiff(run_cfg(src))
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_run_tiff_with_ncsu(
    ncsu_tif: Path,
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    actual = tmp_path.joinpath(ncsu_tif.with_suffix(JPEG_FILE_EXT).name)
    run_tiff(run_cfg(ncsu_tif, output=output_cfg(path=tmp_path)))
    assert actual.exists()
    kind = filetype.guess(str(actual))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(actual))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (500, 500, 3)
    assert np.any(jpeg_img)


def test_expand_sources(
    output_cfg: Callable[..., Output],
    run_cfg: Callable[..., Configuration],
    tmp_path: Path,
):
    paths = [
        tmp_path.joinpath("dst1.tif"),
        tmp_path.joinpath("dst2.dm3"),
        tmp_path.joinpath("dst3.dm4"),
    ]
    actual = expand_sources(
        paths,
        output_cfg(),
        True,
    )
    assert len(actual) == 3
    assert actual[0] == run_cfg(paths[0])
    assert actual[1] == run_cfg(paths[1])
    assert actual[2] == run_cfg(paths[2])


def test_expand_sources_with_directories(
    output_cfg: Callable[..., Output], tmp_path: Path
):
    dir1 = tmp_path.joinpath("dst1")
    os.makedirs(dir1, exist_ok=True)
    file1 = dir1.joinpath("1.tif")
    file2 = dir1.joinpath("2.dm3")
    file3 = dir1.joinpath("3.dm4")
    open(file1, "a").close()
    open(file2, "a").close()
    open(file3, "a").close()
    file4 = tmp_path.joinpath("dst2.dm3")
    file5 = tmp_path.joinpath("dst3.dm4")
    paths = [dir1, file4, file5]
    expected = [file1, file2, file3, file4, file5]
    actual = expand_sources(
        paths,
        output_cfg(),
        True,
    )
    assert len(actual) == 5
    assert actual[0].src in expected
    assert actual[1].src in expected
    assert actual[2].src in expected
    assert actual[3].src in expected
    assert actual[4].src in expected


def test_expand_sources_with_multiple_folders(
    output_cfg: Callable[..., Output], tmp_path: Path
):
    dir1 = tmp_path.joinpath("dst1")
    os.makedirs(dir1, exist_ok=True)
    file1 = dir1.joinpath("1.tif")
    file2 = dir1.joinpath("2.dm3")
    file3 = dir1.joinpath("3.dm4")
    open(file1, "a").close()
    open(file2, "a").close()
    open(file3, "a").close()
    dir2 = tmp_path.joinpath("dst2")
    os.makedirs(dir2, exist_ok=True)
    file4 = dir2.joinpath("4.tif")
    file5 = dir2.joinpath("5.dm3")
    file6 = dir2.joinpath("6.dm4")
    open(file4, "a").close()
    open(file5, "a").close()
    open(file6, "a").close()
    paths = [dir1, dir2]
    expected = [file1, file2, file3, file4, file5, file6]
    actual = expand_sources(
        paths,
        output_cfg(),
        True,
    )
    assert len(actual) == 6
    assert actual[0].src in expected
    assert actual[1].src in expected
    assert actual[2].src in expected
    assert actual[3].src in expected
    assert actual[4].src in expected
    assert actual[5].src in expected


def test_run_with_darwin(dm4: Path, mocker: MockerFixture, tmp_path: Path):
    def mock_platform_system() -> str:
        return "Darwin"

    _ = mocker.patch("platform.system", mock_platform_system)

    def mock_run(cfg: Configuration):
        _ = cfg

    _ = mocker.patch("tsio.cli.run", mock_run)

    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(dm4)])
    assert result.exit_code == 0


def test_run_with_linux(dm4: Path, mocker: MockerFixture, tmp_path: Path):
    def mock_platform_system() -> str:
        return "Linux"

    _ = mocker.patch("platform.system", mock_platform_system)

    def mock_run(cfg: Configuration):
        _ = cfg

    _ = mocker.patch("tsio.cli.run", mock_run)

    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(dm4)])
    assert result.exit_code == 0


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout


def test_app_dcm(dcm: Path, tmp_path: Path):
    dst = tmp_path.joinpath(dcm.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(dcm)])
    assert result.exit_code == 0
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_app_dcm_fails(dcm: Path, tmp_path: Path):
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", "-b", "16", str(dcm)])
    assert result.exit_code == 1


def test_app_dm(dm4: Path, tmp_path: Path):
    dst = tmp_path.joinpath(dm4.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(dm4)])
    assert result.exit_code == 0
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1024, 1024, 3)
    assert np.any(jpeg_img)


def test_app_dm_fails(dm4: Path, tmp_path: Path):
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", "-b", "16", str(dm4)])
    assert result.exit_code == 1


def test_app_all_dm_as_files(dm3: Path, dm4: Path, tmp_path: Path):
    dst_dm3 = tmp_path.joinpath(dm3.name).with_suffix(JPEG_FILE_EXT)
    dst_dm4 = tmp_path.joinpath(dm4.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(dm3), str(dm4)])
    assert result.exit_code == 0
    assert dst_dm3.exists()
    kind = filetype.guess(str(dst_dm3))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst_dm3))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1024, 1024, 3)
    assert np.any(jpeg_img)
    assert dst_dm4.exists()
    kind = filetype.guess(str(dst_dm4))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst_dm4))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1024, 1024, 3)
    assert np.any(jpeg_img)


def test_app_emd(mocker: MockerFixture, single_image_emd: Path, tmp_path: Path):
    src = single_image_emd

    def mock_write_emd(*args: Any, **kwargs: Any):
        _ = args
        _ = kwargs

    _ = mocker.patch("tsio.cli.run_emd", mock_write_emd)

    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(src)])
    assert result.exit_code == 0


def test_app_emd_fails(mocker: MockerFixture, single_image_emd: Path, tmp_path: Path):
    src = single_image_emd

    def mock_write_emd(*args: Any, **kwargs: Any):
        _ = args
        _ = kwargs

    _ = mocker.patch("tsio.cli.run_emd", mock_write_emd)

    result = runner.invoke(app, ["-o", str(tmp_path), "-S", "-b", "16", str(src)])
    assert result.exit_code == 1


def test_app_png_8bit_gray(black_8bit_gray_png: Path, tmp_path: Path):
    dst = tmp_path.joinpath(black_8bit_gray_png.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(black_8bit_gray_png)])
    assert result.exit_code == 0
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_app_png_16bit(black_16bit_gray_png: Path, tmp_path: Path):
    dst = tmp_path.joinpath(black_16bit_gray_png.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(black_16bit_gray_png)])
    assert result.exit_code == 0
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_app_png_white_8bit_rgba(white_8bit_rgba_png: Path, tmp_path: Path):
    dst = tmp_path.joinpath(white_8bit_rgba_png.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["-o", str(tmp_path), "-S", str(white_8bit_rgba_png)])
    assert result.exit_code == 0
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert np.all(jpeg_img == 255)


def test_app_png_fails(black_8bit_gray_png: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "-o",
            str(tmp_path),
            "-S",
            "-b",
            "16",
            str(black_8bit_gray_png),
        ],
    )
    assert result.exit_code == 1


def test_app_tiff(black_16bit_gray_single_page_tiff: Path, tmp_path: Path):
    dst = tmp_path.joinpath(black_16bit_gray_single_page_tiff.name).with_suffix(
        JPEG_FILE_EXT
    )
    result = runner.invoke(
        app,
        [
            "-o",
            str(tmp_path),
            "-S",
            str(black_16bit_gray_single_page_tiff),
        ],
    )
    assert result.exit_code == 0
    assert dst.exists()


def test_app_tiff_fails(black_16bit_gray_single_page_tiff: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "-o",
            str(tmp_path),
            "-S",
            "-b",
            "16",
            str(black_16bit_gray_single_page_tiff),
        ],
    )
    assert result.exit_code == 1
