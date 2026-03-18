# tsio: A command line utility for working with various inputs and outputs related to microscopy

[![CI](https://github.com/Theia-Scientific/tsio/actions/workflows/ci.yml/badge.svg)](https://github.com/Theia-Scientific/tsio/actions/workflows/ci.yml)

A Command Line Interface (CLI) application for converting specialized
image file formats to the common jpeg, png, or tiff formats. Currently
supported specialized formats include:

* [dcm] (DICOM) - medical image file format standard
* [dm3/dm4] (DigitalMicrograph) - (S)TEM image file format standard by GATAN

1. [Prerequisites](#prerequisites)
   1. [Python](#prerequisites-python)
      1. [Ubuntu](#prerequisites-python-ubuntu)
      2. [macOS](#prerequisites-python-macos)
   2. [pipx](#prerequisites-pipx)
      1. [Ubuntu](#prerequisites-pipx-ubuntu)
      2. [macOS](#prerequisites-pipx-macos)
2. [Installation](#installation)
   1. [Application](#installation-app)
      1. [pipx](#installation-app-pipx)
      2. [Source](#installation-app-source)
3. [Configuraton](#configuration)
4. [Upgrade](#upgrade)
   1. [Application](#upgrade-app)
      1. [pipx](#upgrade-app-pipx)
      2. [Source](#upgrade-app-source)
5. [Usage](#usage)
6. [License](#license)

## Prerequisites

All of the prerequisites may already be installed and configured by the
superuser, a.k.a. root, of the computer. The prerequisites only need to be
installed and configured once per machine. For example, if [tsyolo] is already
running, then the prerequisites steps can be ignored.

### Python

<a name="prerequisites-python"></a>

The [Python] programming language is needed to run the `tsio` Command Line
Interface (CLI) application and/or use the `tsio` package in other Python
scripts or [Jupyter] notebooks. Both macOS and Ubuntu Linux have the Python
programming language installed, but it is generally reserved for the operating
system (OS) to use and is an older version. It is best practice to install a
newer version that is separate from the system-provided Python version.

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

### Application

<a name="installation-app"></a>

The `tsio` code includes a Python application with a Command Line Interface
(CLI). It can be installed as a standalone application.

#### pipx (recommended)

<a name="installation-app-pipx"></a>

1. Obtain a package key from Theia Scientific personnel. Save it for Step 3.
2. Ensure `pipx` is installed. See the [Prerequisites](#prerequisites).

   ```sh
   $ pipx --version
   1.9.0
   ```

3. Install `tsio` command globally for all users.

   ```sh
   sudo pipx install --global --python python3.11 --piparg=--extra-index-url="https://pypi:<key>@app.envelope.dev/simple/" "tsio"
   ```

4. Verify `tsio` command is available.

   ```sh
   $ tsio --version
   tsio 0.1.0
   ```

#### Source

<a name="installation-app-source"></a>

1. Clone the [tsio] repository.

   ```sh
   git clone https://github.com/Theia-Scientific/tsio.git && cd tsio
   ```

2. Create a virtual environment.

   ```sh
   python3 -m venv .venv
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
   python3 -m pip install --upgrade pip
   ```

5. Locally install the package, utility, and its dependencies. This will create
   the `tsio` command within the virtual environment. This also installs the
   Slack feature for sending notifications to a Slack channel.

   ```sh
   python3 -m pip install -e .
   ```

## Upgrade

### Application

#### pipx (recommended)

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

#### Source

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
   python -m pip install --upgrade -e .
   ```

5. Verify new version.

   ```sh
   $ tsio --version
   tsio 0.1.0
   ```

## Usage

```sh
$ tsio tiff png image.tif
$ ls
image.png image.tif
```

```sh
$ tsio --output new_name.png tiff png image.tif
$ ls
image.tif new_name.png
```

```sh
$ tsio tiff png multi-page.tif
$ ls
multi-page/ multi-page.tif
$ ls multi-page/
0.png 1.png 2.png 3.png 4.png
```

```sh
$ tsio --output /path/to/directory tiff png multi-page.tif
$ ls
multi-page.tif
$ ls /path/to/directory
0.png 1.png 2.png 3.png 4.png
```

```sh
$ tsio dm jpeg ./00001.dm4
$ ls
00001.dm4  00001.jpg
```

## License

Copyright (C) 2026 Theia Scientific, LLC. All rights reserved.

[dcm]: https://en.wikipedia.org/wiki/DICOM
[deadsnakes]: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
[direnv]: https://direnv.net/
[dm3/dm4]: https://www.gatan.com/products/tem-analysis/gatan-microscopy-suite-software
[homebrew]: https://brew.sh/
[pipx]: https://github.com/pypa/pipx
[python]: https://www.python.org/
[tsio]: https://github.com/Theia-Scientific/tsio
[ultralytics]: https://github.com/ultralytics/ultralytics
