#!/usr/bin/env python3

import gdown
import importlib.metadata
import numpy as np
import os
import pytest

from pathlib import Path
from rsciio.digitalmicrograph import file_reader as dm_file_reader
from rsciio.tiff import file_reader as tiff_file_reader, file_writer as tiff_file_writer
from tsio import __app_name__
from tsio.cli import (
    app,
    map_verbosity,
    JPEG_FILE_EXT,
    JPEG_MIME_TYPE,
    OutputFileFormats,
    PNG_FILE_EXT,
    PNG_MIME_TYPE,
    TIFF_FILE_EXT,
    TIFF_MIME_TYPE,
    write,
    write_dm,
    write_tiff,
)
from typer.testing import CliRunner
from typing import Dict, List

runner = CliRunner()


@pytest.fixture
def blank_16bit_image() -> np.ndarray:
    return np.zeros((1, 256, 256), dtype=np.uint16)


@pytest.fixture
def random_16bit_multipage_image() -> np.ndarray:
    return np.random.randint(0, 2**12, (64, 301, 219), "uint16")


@pytest.fixture
def blank_16bit_single_page_tiff(blank_16bit_image, tmp_path) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {
        "data": blank_16bit_image,
    }
    tiff_file_writer(str(tif_file), signal)
    return tif_file


@pytest.fixture
def random_multipage_tiff(random_16bit_multipage_image, tmp_path) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    signal = {
        "data": random_16bit_multipage_image,
    }
    tiff_file_writer(str(tif_file), signal)
    return tif_file


@pytest.fixture(scope="session")
def dm3(tmp_path_factory) -> Path:
    url = "https://drive.google.com/uc?id=1BDNBta7cSMUtmb4r5e5gvqgBBhRW6xRq"
    dst = tmp_path_factory.mktemp("data").joinpath("test.dm3")
    gdown.download(url, str(dst), quiet=True)
    return dst


@pytest.fixture(scope="session")
def dm4(tmp_path_factory) -> Path:
    url = "https://drive.google.com/uc?id=1Pkbfnl5-7zVSB1h7JfwLbKy6yOxxMvR-"
    dst = tmp_path_factory.mktemp("data").joinpath("test.dm4")
    gdown.download(url, str(dst), quiet=True)
    return dst


def test_output_file_formats_mime_type():
    assert OutputFileFormats.JPEG.mime_type == JPEG_MIME_TYPE
    assert OutputFileFormats.PNG.mime_type == PNG_MIME_TYPE
    assert OutputFileFormats.TIFF.mime_type == TIFF_MIME_TYPE


def test_output_file_formats_file_ext():
    assert OutputFileFormats.JPEG.file_ext == JPEG_FILE_EXT
    assert OutputFileFormats.PNG.file_ext == PNG_FILE_EXT
    assert OutputFileFormats.TIFF.file_ext == TIFF_FILE_EXT


def test_output_file_formats_supported_bit_depths():
    assert OutputFileFormats.JPEG.supported_bit_depths == ["uint8"]
    assert OutputFileFormats.PNG.supported_bit_depths == ["uint8", "uint16"]
    assert OutputFileFormats.TIFF.supported_bit_depths == ["uint8", "uint16"]


def test_map_verbosity_false():
    actual = map_verbosity(False)
    assert actual == "INFO"


def test_map_verbosity_true():
    actual = map_verbosity(True)
    assert actual == "DEBUG"


def test_write_tiff(blank_16bit_single_page_tiff):
    src = blank_16bit_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        None,
        OutputFileFormats.JPEG,
        True,
        normalize=False,
    )
    assert dst.exists()


def test_write_with_output_tiff(blank_16bit_single_page_tiff, tmp_path):
    src = blank_16bit_single_page_tiff
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        tmp_path,
        OutputFileFormats.JPEG,
        True,
        normalize=False,
    )
    assert dst.exists()


def test_write_with_multiple_pages_tiff(
    random_multipage_tiff, random_16bit_multipage_image
):
    pages_count, *_ = random_16bit_multipage_image.shape
    src = random_multipage_tiff
    src_stem = src.stem
    dst = src.parent.joinpath(src_stem)
    write(
        tiff_file_reader(src, multipage_as_list=True),
        src,
        None,
        OutputFileFormats.JPEG,
        True,
        normalize=False,
    )
    assert dst.exists()
    assert (
        len([name for name in os.listdir(dst) if dst.joinpath(name).is_file()])
        == pages_count
    )


def test_write_dm3(dm3):
    src = dm3
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        dm_file_reader(src),
        src,
        None,
        OutputFileFormats.JPEG,
        True,
        normalize=True,
    )
    assert dst.exists()


def test_write_dm4(dm4):
    src = dm4
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(
        dm_file_reader(src),
        src,
        None,
        OutputFileFormats.JPEG,
        True,
        normalize=True,
    )
    assert dst.exists()


def test_write_dm(dm4):
    src = dm4
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_dm((src, None, OutputFileFormats.JPEG, True))
    assert dst.exists()


def test_write_dm_fails_with_not_implemented(mocker, tmp_path):
    src = tmp_path.joinpath("test.dm4")
    dst = src.with_suffix(JPEG_FILE_EXT)

    def mock_file_reader(*args):
        _ = args
        raise NotImplementedError("Not supported version")

    mocker.patch("rsciio.digitalmicrograph.file_reader", mock_file_reader)
    write_dm((src, None, OutputFileFormats.JPEG, True))
    assert not dst.exists()


def test_write_dm_fails_with_exception(tmp_path):
    src = tmp_path.joinpath("test.dm4")
    dst = src.with_suffix(JPEG_FILE_EXT)
    write_dm((src, None, OutputFileFormats.JPEG, True))
    assert not dst.exists()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout
