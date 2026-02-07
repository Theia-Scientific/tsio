#!/usr/bin/env python3

import importlib.metadata

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


def test_output_file_formats_mime_type():
    assert OutputFileFormats.JPEG.mime_type == JPEG_MIME_TYPE
    assert OutputFileFormats.PNG.mime_type == PNG_MIME_TYPE
    assert OutputFileFormats.TIFF.mime_type == TIFF_MIME_TYPE


def test_output_file_formats_file_ext():
    assert OutputFileFormats.JPEG.file_ext == JPEG_FILE_EXT
    assert OutputFileFormats.PNG.file_ext == PNG_FILE_EXT
    assert OutputFileFormats.TIFF.file_ext == TIFF_FILE_EXT


def test_map_verbosity_false():
    actual = map_verbosity(False)
    assert actual == "INFO"


def test_map_verbosity_true():
    actual = map_verbosity(True)
    assert actual == "DEBUG"


# def test_write(tmp_path):
#     src = tmp_path.joinpath("test.tif")
#     dst = src.with_suffix(JPEG_FILE_EXT)
#     write(reader, src, None, OutputFileFormats.JPEG, True, False)
#     assert dst.exists()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout
