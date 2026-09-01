# tsio: A command line utility for extracting images from microscopy-related files

[![CI](https://github.com/Theia-Scientific/tsio/actions/workflows/ci.yml/badge.svg)](https://github.com/Theia-Scientific/tsio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Theia-Scientific/tsio/graph/badge.svg?token=XXXXXXX)](https://codecov.io/gh/Theia-Scientific/tsio)
![PyPI Version](https://img.shields.io/pypi/v/:tsio)

A Command Line Interface (CLI) application for extracting images from microscopy
files to common image file formats, such as JPEG, PNT, and/or TIFF. Supported
microscopy files include:

* [dcm] (DICOM) - medical image file format standard
* [dm3/dm4] (DigitalMicrograph) - (S)TEM image file format by GATAN
* [emd] (Velox) - (S)TEM image file format by ThermoFisher

1. [Prerequisites](#prerequisites)
   1. [Python](#prerequisites-python)
      1. [Ubuntu](#prerequisites-python-ubuntu)
      2. [macOS](#prerequisites-python-macos)
   2. [pipx](#prerequisites-pipx)
      1. [Ubuntu](#prerequisites-pipx-ubuntu)
      2. [macOS](#prerequisites-pipx-macos)
2. [Installation](#installation)
   1. [pipx](#installation-app-pipx)
   2. [Source](#installation-app-source)
3. [Upgrade](#upgrade)
   1. [pipx](#upgrade-app-pipx)
   2. [Source](#upgrade-app-source)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Prerequisites

All of the prerequisites may already be installed and configured by the
superuser, a.k.a. root, of the computer. The prerequisites only need to be
installed and configured once per machine.

### Python

<a name="prerequisites-python"></a>

The [Python] programming language is needed to run the `tsio` Command Line
Interface (CLI) application and/or use in other Python scripts. Both macOS
and Ubuntu Linux have the Python programming language installed, but it is
generally reserved for the operating system (OS) to use and is an older
version. It is best practice to install a newer version that is separate from
the system-provided Python version.

#### Ubuntu

<a name="prerequisites-python-ubuntu"></a>

1. Add the "[deadsnakes]" Ubuntu Personal Package Archives (PPA).

    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    ```
    
2. Obtain the latest packages from the PPA.

   ```sh
   sudo update
   ```

3. Install Python v3.11 or newer.

   ```sh
   sudo apt install python3.11
   ```

4. Install the `venv` package.

   ```sh
   sudo apt install python3.11-venv
   ```

#### macOS

<a name="prerequisites-python-macos"></a>

1. Install [Homebrew] if it is not already installed.

   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python v3.11 or newer.

   ```sh
   brew install python@3.11
   ```

### pipx

<a name="prerequisites-pipx"></a>

The [pipx] utility enables distribution of Python-based CLI applications, like
`tsio`, to be installed for all users with all of the appropriate dependencies
within an isolated environment. It is the recommended installation for the
`tsio` application.

#### Ubuntu

<a name="prerequisites-pipx-ubuntu"></a>

1. Create a virtual environment for `pipx` and Python v3.11 or newer.

   ```sh
   sudo python3.11 -m venv --upgrade-deps /opt/pipx
   ```

2. Install `pipx` for all users.

   ```sh
   sudo /opt/pipx/bin/pip install pipx
   ```

3. Ensure the `pipx` command is available to all users.

   ```sh
   sudo ln -s /opt/pipx/bin/pipx /usr/local/bin/pipx
   ```

4. Add `pipx` to the `PATH` environment variable.

   ```sh
   pipx ensurepath
   ```

5. Add `pipx` for all users.

   ```sh
   sudo pipx ensurepath --global
   ```

Post-installation, the `pipx` application can be upgraded with the following
command:

```sh
sudo /opt/pipx/bin/pip install --upgrade pipx
```

#### macOS

<a name="prerequisites-pipx-macos"></a>

1. Install `pipx` using [Homebrew].

   ```sh
   brew install pipx
   ```

2. Add `pipx` to the `PATH` environment variable.

   ```sh
   pipx ensurepath
   ```

3. Add `pipx` for all users.

   ```sh
   sudo pipx ensurepath --global
   ```

Post-installation, the `pipx` application can be upgraded with the following
command:

``` sh
brew update && brew upgrade pipx
```

## Installation

The `tsio` code includes a Python application with a Command Line Interface
(CLI). It can be installed as a standalone application.

### pipx (recommended)

<a name="installation-app-pipx"></a>

1. Obtain a package key from Theia Scientific personnel. Save it for Step 3.
2. Ensure `pipx` is installed. See the [Prerequisites](#prerequisites).

   ```sh
   $ pipx --version
   1.9.0
   ```

3. Install `tsio` command globally for all users.

   ```sh
   sudo pipx install --global --python python3.11 "tsio"
   ```

4. Verify `tsio` command is available.

   ```sh
   $ tsio --version
   tsio 0.1.0
   ```

### Source

<a name="installation-app-source"></a>

1. Clone the [tsio] repository.

   ```sh
   git clone https://github.com/Theia-Scientific/tsio.git && cd tsio
   ```

2. Create a virtual environment.

   ```sh
   python3.11 -m venv .venv
   ```

3. Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```

   or if [direnv] is installed:

   ```sh
   cp .envrc.example .envrc
   ```

   followed by:

   ```sh
   direnv allow
   ```

4. Upgrade `pip` to the latest version.

   ```sh
   pip install --upgrade pip
   ```

5. Locally install the package, utility, and its dependencies. This will create
   the `tsio` command within the virtual environment. This also installs the
   Slack feature for sending notifications to a Slack channel.

   ```sh
   pip install -e .
   ```

## Upgrade

### pipx (recommended)

<a name="upgrade-app-pipx"></a>

1. Upgrade the `tsio` application via `pipx`.

   ```sh
   sudo pipx upgrade --global tsio
   ```

2. Verify new version.

   ```sh
   $ tsio --version
   tsio 0.1.0
   ```

### Source

<a name="upgrade-app-source"></a>

1. Navigate to the root of the source tree.

   ```sh
   cd ~/Code/tsio
   ```

2. Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```

   or if [direnv] is installed, the virtual environment will automatically be
   activated.

3. Pull the latest changes on `main`.

   ```sh
   git pull
   ```

4. Upgrade the `tsio` application within the virtual environment.

   ```sh
   pip install --upgrade -e .
   ```

5. Verify new version.

   ```sh
   $ tsio --version
   tsio 0.1.0
   ```

## Usage

Convert a single page TIFF to a JPEG. Creating a JPEG image file is the default
because this is useful for Machine Learning (ML) tools, supported by all web
browsers, and it creates the smallest file, making this the quickest for
obtaining a thumbnail of the microscopy image file.

```sh
$ tsio image.tif
$ ls
image.jpg image.tif
```

Convert a single page TIFF to a PNG. All of the following commands are equivalent.

```sh
# Use the long option.
$ tsio --to png image.tif
$ ls
image.png image.tif

# Use the `=` syntax for specifying an option value.
$ tsio --to=png image.tif
$ ls
image.png image.tif

# Use the short option.
$ tsio -t png image.tif
$ ls
image.png image.tif

# Use the short option with the `=` syntax.
$ tsio -t=png image.tif
$ ls
image.png image.tif
```

Extract all the "frames" from a multi-page TIFF.

```sh
$ tsio multi-page.tif
$ ls
multi-page/ multi-page.tif
$ ls multi-page/
0.jpg 1.jpg 2.jpg 3.jpg 4.jpg

# Use the `-o,--output` option to extract to different location.
$ tsio --output /path/to/directory tiff png multi-page.tif
$ ls
multi-page.tif
$ ls /path/to/directory
0.jpg 1.jpg 2.jpg 3.jpg 4.jpg
```

Extract an image from a DM3 or DM4 file.

```sh
$ tsio ./00001.dm3
$ ls
00001.dm3  00001.jpg

$ tsio ./00001.dm4
$ ls
00001.dm4  00001.jpg
```

Extract all the "frames" from a EMD file.

```sh
$ tsio multiple-frames.emd
$ ls
multiple-frames/ multiple-frames.emd
$ ls multiple-frames/
0.jpg 1.jpg 2.jpg 3.jpg 4.jpg
```

## Contributing

1. Clone this repository.

   ```sh
   git clone https://github.com/Theia-Scientific/tsio && cd tsio
   ```

2. Create a virtual environment.

   ```sh
   python3 -m venv .venv
   ```

3. Activate the virtual environment.

   ```sh
   source .venv/bin/activate
   ```
   
   or if [direnv] is installed, the virtual environment will automatically be
   activated.

4. Upgrade `pip`.

   ```sh
   pip install --upgrade pip
   ```
   
5. Install all the dependencies.

   ```sh
   pip install -e ".[dev]"
   ```

6. Create a local branch.

   ```sh
   git checkout -b feature-awesome-new-feature
   ```

7. Modify the code.
8. Run the tests.

   ```sh
   pytest --color=yes --cov=tsio --cov-report=term-missing
   ```

9. Commit changes to your local branch.

   ```sh
   git add -A && git commit -m "Add new feature"
   ```

10. Push your local branch to GitHub to create a Pull Request (PR).

   ```sh
   git push origin feature-awesome-new-feature
   ```

11. Create a Pull Request (PR) in GitHub.
12. Wait for CI to complete.
13. Add comment to PR that it is ready to review.
14. Wait for review from a maintainer.
15. Address any comments from the reviewer by modifying your local files and
    pushing to the remote branch/PR.
    
    ```sh
    git push origin feature-awesome-new-feature
    ```
    
16. Once the PR is approved, then it will be "Squash and Merge". Congratulations
    on contributing to an open source project, and Thank you!

## License

The `tsio` project is licensed under the [GPL-3.0] license. See the
[LICENSE.txt] file for more information about licensing and copyright.

## Acknowledgments

This material is based upon work supported by the U.S. Department of Energy,
Office of Basic Science and Office of Nuclear Energy under Award Number
DE-SC0021529 and DE-SC0021936, respectively.

[dcm]: https://en.wikipedia.org/wiki/DICOM
[deadsnakes]: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
[direnv]: https://direnv.net/
[dm3/dm4]: https://www.gatan.com/products/tem-analysis/gatan-microscopy-suite-software
[emd]: https://hyperspy.org/rosettasciio/supported_formats/emd.html#emd-fei-format
[gpl-3.0]: https://opensource.org/license/gpl-3.0
[homebrew]: https://brew.sh/
[license.txt]: https://github.com/Theia-Scientific/tsio/blob/main/LICENSE.txt
[pipx]: https://github.com/pypa/pipx
[python]: https://www.python.org/
[tsio]: https://github.com/Theia-Scientific/tsio
