#!/usr/bin/env python3

import cv2
import datetime
import filetype
import gdown
import importlib.metadata
import logging
import numpy as np
import os
import pytest

from pathlib import Path
from pydantic import ValidationError
from pydicom import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian
from rsciio.digitalmicrograph import file_reader as dm_file_reader
from rsciio.image import file_writer as image_file_writer
from rsciio.tiff import file_reader as tiff_file_reader, file_writer as tiff_file_writer
from rsciio.utils import rgb
from tsio import __app_name__
from tsio.cli import (
    app,
    BitDepths,
    Configuration,
    expand_sources,
    map_verbosity,
    JPEG_FILE_EXT,
    JPEG_MIME_TYPE,
    Output,
    OutputFileFormats,
    PNG_FILE_EXT,
    PNG_MIME_TYPE,
    run,
    TIFF_FILE_EXT,
    TIFF_MIME_TYPE,
    write,
    write_dcm,
    write_dm,
    write_emd,
    write_png,
    write_tiff,
)
from typer.testing import CliRunner
from typing import Callable

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
def black_8bit_gray_png(black_8bit_gray_image, tmp_path) -> Path:
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
def black_8bit_rgb_png(black_8bit_rgb_image, tmp_path) -> Path:
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
def black_8bit_rgba_png(black_8bit_rgba_image, tmp_path) -> Path:
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
def black_16bit_gray_png(black_16bit_gray_image, tmp_path) -> Path:
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
def black_16bit_rgb_png(black_16bit_rgb_image, tmp_path) -> Path:
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
def black_16bit_rgba_png(black_16bit_rgba_image, tmp_path) -> Path:
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
def white_8bit_gray_png(white_8bit_gray_image, tmp_path) -> Path:
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
def white_8bit_rgb_png(white_8bit_rgb_image, tmp_path) -> Path:
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
def white_8bit_rgba_png(white_8bit_rgba_image, tmp_path) -> Path:
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
def white_16bit_gray_png(white_16bit_gray_image, tmp_path) -> Path:
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
def white_16bit_rgb_png(white_16bit_rgb_image, tmp_path) -> Path:
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
def white_16bit_rgba_png(white_16bit_rgba_image, tmp_path) -> Path:
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
def random_16bit_multipage_image() -> np.ndarray:
    return np.random.randint(0, 2**12, (64, 301, 219), "uint16")


@pytest.fixture
def black_8bit_gray_single_page_tiff(black_8bit_gray_image, tmp_path) -> Path:
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
def black_8bit_rgb_single_page_tiff(black_8bit_rgb_image, tmp_path) -> Path:
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
def black_8bit_rgba_single_page_tiff(black_8bit_rgba_image, tmp_path) -> Path:
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
def black_16bit_gray_single_page_tiff(black_16bit_gray_image, tmp_path) -> Path:
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
def black_16bit_rgb_single_page_tiff(black_16bit_rgb_image, tmp_path) -> Path:
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
def black_16bit_rgba_single_page_tiff(black_16bit_rgba_image, tmp_path) -> Path:
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
def random_multipage_tiff(random_16bit_multipage_image, tmp_path) -> Path:
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
def heart_png(assets) -> Path:
    return assets.joinpath("heart16.png")


@pytest.fixture
def ncsu_tif(assets) -> Path:
    return assets.joinpath("1138622_small_slice_42.tif")


@pytest.fixture
def v09_loaded_confusion_matrix_png(assets) -> Path:
    return assets.joinpath("v09_loaded_confusion_matrix.png")


@pytest.fixture
def dcm(tmp_path, black_8bit_gray_image) -> Path:
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
def dm3(tmp_assets) -> Path:
    dst = tmp_assets.joinpath("2.dm3")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1BDNBta7cSMUtmb4r5e5gvqgBBhRW6xRq"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def dm4(tmp_assets) -> Path:
    dst = tmp_assets.joinpath("1.dm4")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1Pkbfnl5-7zVSB1h7JfwLbKy6yOxxMvR-"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def emd_single_image(tmp_assets) -> Path:
    dst = tmp_assets.joinpath("3.emd")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1Z-aJUxQdpzd4v5ptOZvIY5Q8EYYSGi7L"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def emd_multiple_images(tmp_assets) -> Path:
    dst = tmp_assets.joinpath("4.emd")
    if not dst.exists():
        url = "https://drive.google.com/uc?id=1GDB9hvN1FAULy1JwQN0THL7fAs4BbTlf"
        gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture
def output_cfg() -> Callable[[BitDepths, OutputFileFormats, Path], Output]:
    def _make_output_cfg(
        bit_depth=BitDepths.EIGHT, format=OutputFileFormats.JPEG, path=None
    ) -> Output:
        return Output(bit_depth=bit_depth, format=format, path=path)

    return _make_output_cfg


def test_bit_depths_type():
    assert BitDepths.EIGHT.type == "uint8"
    assert BitDepths.SIXTEEN.type == "uint16"


def test_bit_depths_max_pixel_intensity():
    assert BitDepths.EIGHT.max_pixel_intensity == 255
    assert BitDepths.SIXTEEN.max_pixel_intensity == 65535


def test_output_file_formats_is_alpha_supported():
    assert not OutputFileFormats.JPEG.is_alpha_supported
    assert OutputFileFormats.PNG.is_alpha_supported
    assert OutputFileFormats.TIFF.is_alpha_supported


def test_output_file_formats_is_gray_supported():
    assert not OutputFileFormats.JPEG.is_gray_supported
    assert OutputFileFormats.PNG.is_gray_supported
    assert OutputFileFormats.TIFF.is_gray_supported


def test_output_file_formats_mime_type():
    assert OutputFileFormats.JPEG.mime_type == JPEG_MIME_TYPE
    assert OutputFileFormats.PNG.mime_type == PNG_MIME_TYPE
    assert OutputFileFormats.TIFF.mime_type == TIFF_MIME_TYPE


def test_output_file_formats_file_ext():
    assert OutputFileFormats.JPEG.file_ext == JPEG_FILE_EXT
    assert OutputFileFormats.PNG.file_ext == PNG_FILE_EXT
    assert OutputFileFormats.TIFF.file_ext == TIFF_FILE_EXT


def test_output_destination_with_none_path(tmp_path, output_cfg):
    assert output_cfg().destination(tmp_path) == tmp_path.resolve().parent


def test_output_destination_with_path(tmp_path, output_cfg):
    assert output_cfg(path=tmp_path).destination(tmp_path) == tmp_path


def test_output_is_gray(
    black_8bit_gray_image,
    black_16bit_gray_image,
    black_8bit_rgb_image,
    black_16bit_rgb_image,
    black_8bit_rgba_image,
    black_16bit_rgba_image,
):
    assert Output.is_gray(black_8bit_gray_image)
    assert Output.is_gray(black_16bit_gray_image)
    assert not Output.is_gray(black_8bit_rgb_image)
    assert not Output.is_gray(black_16bit_rgb_image)
    assert not Output.is_gray(black_8bit_rgba_image)
    assert not Output.is_gray(black_16bit_rgba_image)


def test_output_cast_eight_gray(black_16bit_gray_image, output_cfg, tmp_path):
    img = output_cfg(path=tmp_path).cast(black_16bit_gray_image)
    assert img.dtype.name == "uint8"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_eight_rgb(black_16bit_rgb_image, output_cfg, tmp_path):
    img = output_cfg(path=tmp_path).cast(black_16bit_rgb_image)
    assert img.dtype.name == "uint8"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_eight_rgba(black_16bit_rgba_image, output_cfg, tmp_path):
    img = output_cfg(path=tmp_path).cast(black_16bit_rgba_image)
    assert img.dtype.name == "uint8"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_sixteen_gray(black_16bit_gray_image, output_cfg, tmp_path):
    img = output_cfg(
        bit_depth=BitDepths.SIXTEEN, format=OutputFileFormats.PNG, path=tmp_path
    ).cast(black_16bit_gray_image)
    assert img.dtype.name == "uint16"
    assert img.shape == (256, 256)
    assert not np.any(img)


def test_output_cast_sixteen_rgb(black_16bit_rgb_image, output_cfg, tmp_path):
    img = output_cfg(
        bit_depth=BitDepths.SIXTEEN, format=OutputFileFormats.PNG, path=tmp_path
    ).cast(black_16bit_rgb_image)
    assert img.dtype.name == "uint16"
    assert img.shape == (256, 256, 3)
    assert not np.any(img)


def test_output_cast_sixteen_rgba(black_16bit_rgba_image, output_cfg, tmp_path):
    img = output_cfg(
        bit_depth=BitDepths.SIXTEEN, format=OutputFileFormats.PNG, path=tmp_path
    ).cast(black_16bit_rgba_image)
    assert img.dtype.name == "uint16"
    assert img.shape == (256, 256, 4)
    assert not np.any(img)


def test_output_validation():
    with pytest.raises(ValidationError):
        Output(bit_depth=BitDepths.SIXTEEN, format=OutputFileFormats.JPEG, path=None)


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


def test_write_tiff_with_black_8bit_gray(black_8bit_gray_single_page_tiff, output_cfg):
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


def test_write_tiff_with_black_8bit_rgb(black_8bit_rgb_single_page_tiff, output_cfg):
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


def test_write_tiff_with_black_8bit_rgba(black_8bit_rgba_single_page_tiff, output_cfg):
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
    black_16bit_gray_single_page_tiff, output_cfg
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
    black_16bit_gray_single_page_tiff, output_cfg
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
    output_cfg, random_multipage_tiff, random_16bit_multipage_image
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


def test_write_with_dm3(dm3, output_cfg, tmp_path):
    src = dm3
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    write(
        dm_file_reader(src),
        src,
        output_cfg(path=tmp_path),
        True,
    )
    assert dst.exists()


def test_write_with_dm4(dm4, output_cfg, tmp_path):
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


def test_write_dcm(dcm, output_cfg, tmp_path):
    src = dcm
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    write_dcm(
        Configuration(
            output=output_cfg(path=tmp_path),
            silent=True,
            src=src,
        )
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


def test_write_dm(dm4, output_cfg, tmp_path):
    src = dm4
    dst = tmp_path.joinpath(src.with_suffix(JPEG_FILE_EXT).name)
    write_dm(
        Configuration(
            output=output_cfg(path=dst),
            silent=True,
            src=src,
        )
    )
    assert dst.exists()


def test_write_dm_fails_with_not_implemented(mocker, output_cfg, tmp_path):
    src = tmp_path.joinpath("test.dm4")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args, **kwargs):
        _ = args
        _ = kwargs

        raise NotImplementedError("Not supported version")

    mocker.patch("rsciio.digitalmicrograph.file_reader", mock_file_reader)
    write_dm(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
    )
    assert not dst.exists()


def test_write_dm_fails_with_exception(output_cfg, tmp_path):
    src = tmp_path.joinpath("test.dm4")
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_dm(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
    )
    assert not dst.exists()


def test_write_emd_single_image(emd_single_image, output_cfg):
    src = emd_single_image
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_emd(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_emd_multiple_images(emd_multiple_images, output_cfg, tmp_path):
    src = emd_multiple_images
    dst = tmp_path.joinpath(src.stem)
    write_emd(
        Configuration(
            output=output_cfg(path=tmp_path),
            silent=True,
            src=src,
        )
    )
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


def test_write_emd_with_no_image_data(mocker, output_cfg, tmp_path):
    src = tmp_path.joinpath("test.emd")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args, **kwargs):
        _ = args
        _ = kwargs

        return []

    mocker.patch("rsciio.emd.file_reader", mock_file_reader)
    write_emd(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
    )
    assert not dst.exists()


def test_write_emd_with_no_data_field(mocker, output_cfg, tmp_path):
    src = tmp_path.joinpath("test.emd")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args, **kwargs):
        _ = args
        _ = kwargs

        return [{"axes": []}]

    mocker.patch("rsciio.emd.file_reader", mock_file_reader)
    write_emd(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
    )
    assert not dst.exists()


def test_write_emd_fails_with_exception(mocker, output_cfg, tmp_path):
    src = tmp_path.joinpath("test.emd")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args, **kwargs):
        _ = args
        _ = kwargs

        raise Exception("Test Exception")

    mocker.patch("rsciio.emd.file_reader", mock_file_reader)
    write_emd(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
    )
    assert not dst.exists()


def test_write_png_8bit_gray(black_8bit_gray_png, output_cfg):
    src = black_8bit_gray_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_png_8bit_rgb(black_8bit_rgb_png, output_cfg):
    src = black_8bit_rgb_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_png_8bit_rgba(black_8bit_rgba_png, output_cfg):
    src = black_8bit_rgba_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_png_16bit_gray(black_16bit_gray_png, output_cfg):
    src = black_16bit_gray_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_png_16bit_rgb(black_16bit_rgb_png, output_cfg):
    src = black_16bit_rgb_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_png_16bit_rgba(black_16bit_rgba_png, output_cfg):
    src = black_16bit_rgba_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_8bit_grayscale_png_to_8bit_tiff(
    black_8bit_gray_png, output_cfg, tmp_path
):
    src = black_8bit_gray_png
    dst = src.with_suffix(TIFF_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(format=OutputFileFormats.TIFF),
            silent=True,
            src=src,
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
    assert not np.any(tiff_img)


def test_write_8bit_grayscale_png_to_16bit_tiff(
    black_8bit_gray_image, output_cfg, tmp_path
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
    write_png(
        Configuration(
            output=output_cfg(
                bit_depth=BitDepths.SIXTEEN, format=OutputFileFormats.TIFF
            ),
            silent=True,
            src=src,
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


def test_write_16bit_grayscale_png_to_8bit_tiff(black_16bit_gray_png, output_cfg):
    src = black_16bit_gray_png
    dst = src.with_suffix(TIFF_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(format=OutputFileFormats.TIFF),
            silent=True,
            src=src,
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
    assert not np.any(tiff_img)


def test_write_16bit_grayscale_png_to_16bit_tiff(black_16bit_gray_png, output_cfg):
    src = black_16bit_gray_png
    dst = src.with_suffix(TIFF_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(
                bit_depth=BitDepths.SIXTEEN, format=OutputFileFormats.TIFF
            ),
            silent=True,
            src=src,
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


def test_write_heart_png_to_8bit_gray_tiff(heart_png, output_cfg, tmp_path):
    src = heart_png
    dst = tmp_path.joinpath(src.name).with_suffix(TIFF_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(
                bit_depth=BitDepths.EIGHT, format=OutputFileFormats.TIFF, path=tmp_path
            ),
            silent=True,
            src=src,
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


def test_write_heart_png_to_16bit_gray_tiff(heart_png, output_cfg, tmp_path):
    src = heart_png
    dst = tmp_path.joinpath(src.name).with_suffix(TIFF_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(
                bit_depth=BitDepths.SIXTEEN,
                format=OutputFileFormats.TIFF,
                path=tmp_path,
            ),
            silent=True,
            src=src,
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


def test_write_confusion_matrix_png_to_jpeg(
    v09_loaded_confusion_matrix_png, output_cfg, tmp_path
):
    src = v09_loaded_confusion_matrix_png
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(path=tmp_path),
            silent=True,
            src=src,
        )
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (1606, 1840, 3)
    assert np.any(jpeg_img)


def test_write_white_8bit_rgba_png(white_8bit_rgba_png, output_cfg):
    src = white_8bit_rgba_png
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_png(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
    )
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert np.all(jpeg_img, where=255)


def test_write_tiff(black_16bit_gray_single_page_tiff, output_cfg):
    src = black_16bit_gray_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_tiff(
        Configuration(
            output=output_cfg(),
            silent=True,
            src=src,
        )
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


def test_write_tiff_with_nontiff(
    black_8bit_gray_image,
    black_16bit_gray_image,
    caplog,
    output_cfg,
    tmp_path,
):
    png_file = tmp_path.joinpath("image.png")
    signal = {"data": black_8bit_gray_image, "axes": {}}
    image_file_writer(str(png_file), signal)
    tif_file = tmp_path.joinpath("image.tif")
    signal = {
        "data": black_16bit_gray_image,
    }
    tiff_file_writer(str(tif_file), signal)
    with caplog.at_level(logging.WARNING):
        write_tiff(
            Configuration(
                output=output_cfg(path=tmp_path),
                silent=False,
                src=png_file,
            )
        )
    write_tiff(
        Configuration(
            output=Output(
                bit_depth=BitDepths.EIGHT,
                format=OutputFileFormats.JPEG,
                path=tmp_path,
            ),
            silent=True,
            src=tif_file,
        )
    )
    dst = tif_file.with_suffix(JPEG_FILE_EXT)
    assert "file is not a TIFF, skipped." in caplog.text
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert not np.any(jpeg_img)


def test_write_tiff_with_ncsu(ncsu_tif, output_cfg, tmp_path):
    actual = tmp_path.joinpath(ncsu_tif.with_suffix(JPEG_FILE_EXT).name)
    write_tiff(
        Configuration(
            output=output_cfg(path=tmp_path),
            silent=True,
            src=ncsu_tif,
        )
    )
    assert actual.exists()
    kind = filetype.guess(str(actual))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(actual))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (500, 500, 3)
    assert np.any(jpeg_img)


def test_expand_sources(output_cfg, tmp_path):
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
    assert actual[0] == Configuration(
        output=output_cfg(),
        silent=True,
        src=paths[0],
    )
    assert actual[1] == Configuration(
        output=output_cfg(),
        silent=True,
        src=paths[1],
    )
    assert actual[2] == Configuration(
        output=output_cfg(),
        silent=True,
        src=paths[2],
    )


def test_expand_sources_with_directories(output_cfg, tmp_path):
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


def test_run_with_darwin(dm4, mocker, output_cfg):
    def mock_platform_system() -> str:
        return "Darwin"

    mocker.patch("platform.system", mock_platform_system)

    def mock_write(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    run(
        mock_write,
        [
            Configuration(
                output=output_cfg(),
                silent=True,
                src=dm4,
            )
        ],
    )


def test_run_with_linux(dm4, mocker, output_cfg):
    def mock_platform_system() -> str:
        return "Linux"

    mocker.patch("platform.system", mock_platform_system)

    def mock_write(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    run(
        mock_write,
        [
            Configuration(
                output=output_cfg(),
                silent=True,
                src=dm4,
            )
        ],
    )


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout


def test_app_dcm(dcm, tmp_path):
    dst = tmp_path.joinpath(dcm.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["dcm", "-o", str(tmp_path), "-S", "jpeg", str(dcm)])
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


def test_app_dcm_fails(dcm, tmp_path):
    result = runner.invoke(
        app, ["dcm", "-o", str(tmp_path), "-S", "-b", "16", "jpeg", str(dcm)]
    )
    assert result.exit_code == 1


def test_app_dm(dm4, tmp_path):
    dst = tmp_path.joinpath(dm4.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(app, ["dm", "-o", str(tmp_path), "-S", "jpeg", str(dm4)])
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


def test_app_dm_fails(dm4, tmp_path):
    result = runner.invoke(
        app, ["dm", "-o", str(tmp_path), "-S", "-b", "16", "jpeg", str(dm4)]
    )
    assert result.exit_code == 1


def test_app_all_dm_as_files(dm3, dm4, tmp_path):
    dst_dm3 = tmp_path.joinpath(dm3.name).with_suffix(JPEG_FILE_EXT)
    dst_dm4 = tmp_path.joinpath(dm4.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(
        app, ["dm", "-o", str(tmp_path), "-S", "jpeg", str(dm3), str(dm4)]
    )
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


def test_app_all_dm_as_directory(dm3, dm4, tmp_assets, tmp_path):
    dst_dm3 = tmp_path.joinpath(dm3.name).with_suffix(JPEG_FILE_EXT)
    dst_dm4 = tmp_path.joinpath(dm4.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(
        app, ["dm", "-o", str(tmp_path), "-S", "jpeg", str(tmp_assets)]
    )
    assert result.exit_code == 0
    assert dst_dm3.exists()
    assert dst_dm4.exists()


def test_app_emd(mocker, emd_single_image, tmp_path):
    src = emd_single_image

    def mock_write_emd(*args, **kwargs):
        _ = args
        _ = kwargs

    mocker.patch("tsio.cli.write_emd", mock_write_emd)

    result = runner.invoke(app, ["emd", "-o", str(tmp_path), "-S", "jpeg", str(src)])
    assert result.exit_code == 0


def test_app_emd_fails(mocker, emd_single_image, tmp_path):
    src = emd_single_image

    def mock_write_emd(*args, **kwargs):
        _ = args
        _ = kwargs

    mocker.patch("tsio.cli.write_emd", mock_write_emd)

    result = runner.invoke(
        app, ["emd", "-o", str(tmp_path), "-S", "-b", "16", "jpeg", str(src)]
    )
    assert result.exit_code == 1


def test_app_png_8bit_gray(black_8bit_gray_png, tmp_path):
    dst = tmp_path.joinpath(black_8bit_gray_png.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(
        app, ["png", "-o", str(tmp_path), "-S", "jpeg", str(black_8bit_gray_png)]
    )
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


def test_app_png_16bit(black_16bit_gray_png, tmp_path):
    dst = tmp_path.joinpath(black_16bit_gray_png.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(
        app, ["png", "-o", str(tmp_path), "-S", "jpeg", str(black_16bit_gray_png)]
    )
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


def test_app_png_white_8bit_rgba(white_8bit_rgba_png, tmp_path):
    dst = tmp_path.joinpath(white_8bit_rgba_png.name).with_suffix(JPEG_FILE_EXT)
    result = runner.invoke(
        app, ["png", "-o", str(tmp_path), "-S", "jpeg", str(white_8bit_rgba_png)]
    )
    assert result.exit_code == 0
    assert dst.exists()
    kind = filetype.guess(str(dst))
    assert kind is not None
    assert kind.mime == "image/jpeg"
    jpeg_img = cv2.imread(str(dst))
    assert jpeg_img is not None
    assert jpeg_img.dtype.name == "uint8"
    assert jpeg_img.shape == (256, 256, 3)
    assert np.all(jpeg_img, where=1)


def test_app_png_fails(black_8bit_gray_png, tmp_path):
    result = runner.invoke(
        app,
        [
            "png",
            "-o",
            str(tmp_path),
            "-S",
            "-b",
            "16",
            "jpeg",
            str(black_8bit_gray_png),
        ],
    )
    assert result.exit_code == 1


def test_app_tiff(black_16bit_gray_single_page_tiff, tmp_path):
    dst = tmp_path.joinpath(black_16bit_gray_single_page_tiff.name).with_suffix(
        JPEG_FILE_EXT
    )
    result = runner.invoke(
        app,
        [
            "tiff",
            "-o",
            str(tmp_path),
            "-S",
            "jpeg",
            str(black_16bit_gray_single_page_tiff),
        ],
    )
    assert result.exit_code == 0
    assert dst.exists()


def test_app_tiff_fails(black_16bit_gray_single_page_tiff, tmp_path):
    result = runner.invoke(
        app,
        [
            "tiff",
            "-o",
            str(tmp_path),
            "-S",
            "-b",
            "16",
            "jpeg",
            str(black_16bit_gray_single_page_tiff),
        ],
    )
    assert result.exit_code == 1
