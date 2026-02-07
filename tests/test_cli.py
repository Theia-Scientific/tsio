#!/usr/bin/env python3

import importlib.metadata
import numpy as np
import os
import pytest

from pathlib import Path
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
)
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def blank_16bit_image() -> np.ndarray:
    return np.zeros((1, 256, 256), dtype=np.uint16)


@pytest.fixture
def random_16bit_multipage_image() -> np.ndarray:
    return np.random.randint(0, 2**12, (64, 301, 219), "uint16")


@pytest.fixture
def blank_single_page_tiff(blank_16bit_image, tmp_path) -> Path:
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


def test_write(blank_single_page_tiff):
    src = blank_single_page_tiff
    dst = src.with_suffix(JPEG_FILE_EXT)
    write(tiff_file_reader, src, None, OutputFileFormats.JPEG, True, False)
    assert dst.exists()


def test_write_with_output(blank_single_page_tiff, tmp_path):
    src = blank_single_page_tiff
    dst = tmp_path.joinpath(src.name).with_suffix(JPEG_FILE_EXT)
    write(tiff_file_reader, src, tmp_path, OutputFileFormats.JPEG, True, False)
    assert dst.exists()


def test_write_with_multiple_pages(random_multipage_tiff, random_16bit_multipage_image):
    pages_count, *_ = random_16bit_multipage_image.shape
    src = random_multipage_tiff
    src_stem = src.stem
    dst = src.parent.joinpath(src_stem)
    write(tiff_file_reader, src, None, OutputFileFormats.JPEG, True, False)
    assert dst.exists()
    assert (
        len([name for name in os.listdir(dst) if dst.joinpath(name).is_file()])
        == pages_count
    )


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout
