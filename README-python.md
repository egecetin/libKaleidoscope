<div align="center">

<img src="https://raw.githubusercontent.com/egecetin/libKaleidoscope/master/doc/images/logo-black.png" alt="" width="850"/>
<br>

A library to create kaleidoscope effect on images. You can build on all platforms using CMake.

![GitHub](https://img.shields.io/github/license/egecetin/libKaleidoscope?style=for-the-badge)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/egecetin/libKaleidoscope/pre-commit.yml?branch=master&label=pre-commit&logo=precommit&logoColor=white&style=for-the-badge)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/egecetin/libKaleidoscope/codeql-analysis.yml?branch=master&label=CodeQL&logo=github&style=for-the-badge)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/egecetin/libKaleidoscope/os-builds.yml?branch=master&label=Build&logo=github&logoColor=white&style=for-the-badge)
![Codecov](https://img.shields.io/codecov/c/github/egecetin/libkaleidoscope?logo=codecov&logoColor=white&style=for-the-badge&token=70EJQJRRBH)
![Codacy grade](https://img.shields.io/codacy/grade/b6c3a6abeeb34c2e8aa67aaeb8bd2982?logo=codacy&style=for-the-badge)


![C Badge](https://img.shields.io/badge/C-%23555555?style=for-the-badge&logo=c&logoColor=white)
![C++ Badge](https://img.shields.io/badge/C%2B%2B-%23f34b7d?style=for-the-badge&logo=cplusplus&logoColor=white)
![Python Badge](https://img.shields.io/badge/Python-%233572A5?style=for-the-badge&logo=python&logoColor=white)
![CUDA Badge](https://img.shields.io/badge/CUDA-%233A4E3A?style=for-the-badge&logo=nvidia&logoColor=white)
</div>

The library is written in C language so you can use Foreign Function Interface (FFI) to call functions from your favorite programming language. You can download from python package from PyPI. It also has C++ header only library to provide easier interface for C++ users and CUDA support for people who have doubts about performance. Check for mathematical explanation of the kaleidoscope effect from my [webpage](https://egecetin.github.io/Projects/kaleidoscope)

## Supported Languages

- C : Main programming language
- C++ : Header only binding for easier usage
- Python : Bindings using Cython
- CUDA : For GPU computing

## Install for Python

```
pip install LibKaleidoscope
```

## Building

Use the following commands,

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

If you want to enable CUDA backend,

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DKALEIDOSCOPE_ENABLE_CUDA=ON ..
cmake --build . --parallel
```

There is no direct dependency for libjpeg-turbo inside from the library. It is just for test and demonstration purposes. If you don't want to install/compile just disable command line tool compilation with ``-DKALEIDOSCOPE_ENABLE_CMD_TOOL=OFF``

## Usage

The library has a simple usage and you need only three functions to use it. Check the sample usage at ``src/kaleidoscope-cmd.c``

- Initialization of the transformation matrix: ``int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, double scaleDown)``
- Processing image (Can be used multiple times if the input images have same dimensions): ``void processKaleidoscope(KaleidoscopeHandle *handler, double k, unsigned char *imgIn, unsigned char *imgOut)``
- Deinitialization of the transformation matrix: ``void deInitKaleidoscope(KaleidoscopeHandle *handler)``

Alternatively you can directly use the command line program to create kaleidoscope effect with ``./kaleidoscope-cmd <Input Image Path> <Output Image Path> <N>``. You can see an example below for ``N=8``
<div align="center">
    <img src="https://raw.githubusercontent.com/egecetin/libKaleidoscope/master/doc/images/ac-synin.jpg" width="425"/> <img src="https://raw.githubusercontent.com/egecetin/libKaleidoscope/master/doc/images/ac-synin-out.jpg" width="425"/>
    <br>
    Image source: AC Valhalla
</div>

For C++ and CUDA usage check the unit tests at ``tests/processingTest.cpp`` and ``tests/processingTest.cu``. It is very easy! Just include the header and construct the ``Kaleidoscope`` class from ``kalos`` namespace.

- For C++ header only binding,

```
#include <kaleidoscope.hpp>

int main()
{
    kalos::Kaleidoscope handler(n, width, height, nComponents, scaleDown, k);

    /* ... */

    handler.processImage(inData, outData, nPixel);

    /* ... */

    return 0;
}
```

- For CUDA backend,

```
#include <cuda/kaleidoscope.cuh>

int main()
{
    kalos::cuda::Kaleidoscope handler(n, width, height, nComponents, scaleDown, k);

    /* ... */

    // Make sure inData and outData is device allocated!
    handler.processImage(inData, outData, nPixel);

    /* ... */

    return 0;
}
```

## Benchmark

It is really fast! On a Intel i7-11800H CPU it achieves,

- ~65 FPS for 4K UHD (3840 x 2160)
- ~265 FPS for Full HD (1920 x 1080)
- ~640 FPS for 720p (1280 x 720)
- ~1350 FPS for 576p (720 x 576)

resolution images. The performance estimation can be seen at the below

<div align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="doc/images/performance-white.png"/>
    <img src="doc/images/performance-black.png" alt="" width="850"/>
    </picture>
</div>

$$ FPS = a\text{ }e^{b\text{ }nPixels}+c\text{ }e^{d\text{ }nPixels} $$

$$ a = 2492 \text{, } b = -2.165\text{ }10^{-6} \text{, } c = 364.9 \text{, } d = -2.08\text{ }10^{-7} $$

If you want to benchmark code on your system make sure you configured with ```-DCMAKE_BUILD_TYPE=Release``` and use this command,

```./kaleidoscope-cmd <Input Image Path> <Output Image Path> <N> <Number of loop>```
