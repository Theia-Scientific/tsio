# tsio: A command line utility for working with various inputs and outputs related to microscopy

[![CI](https://github.com/Theia-Scientific/tsio/actions/workflows/ci.yml/badge.svg)](https://github.com/Theia-Scientific/tsio/actions/workflows/ci.yml)

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

Copyright (C) 2025 Theia Scientific, LLC. All rights reserved.

