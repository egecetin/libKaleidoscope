<div align="center">

# Kaleidoscope Library
A library to create kaleidoscope effect on images. You can build on all platforms using CMake.

![GitHub](https://img.shields.io/badge/Language-C-informational)
![GitHub](https://img.shields.io/github/license/egecetin/libKaleidoscope)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/egecetin/libKaleidoscope/pre-commit?label=pre-commit&logo=precommit&logoColor=white)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/egecetin/libKaleidoscope/CodeQL?label=CodeQL&logo=github)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/egecetin/libKaleidoscope/Linux?label=Linux&logo=linux&logoColor=white)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/egecetin/libKaleidoscope/MacOS?label=MacOS&logo=apple&logoColor=white)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/egecetin/libKaleidoscope/Windows?label=Windows&logo=windows&logoColor=white)
![codecov](https://codecov.io/gh/egecetin/libKaleidoscope/branch/master/graph/badge.svg?token=70EJQJRRBH)
</div>

## Building

Use the following commands,

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

## Usage

The library has a simple usage and you need only three functions to use it. Check the sample usage at ```src/kaleidoscope-cmd.c```

- Initialization of the transformation matrix: ```int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, double scaleDown)```
- Processing image (Can be used multiple times if the input images have same dimensions): ```void processKaleidoscope(KaleidoscopeHandle *handler, double k, ImageData *imgIn, ImageData *imgOut)```
- Deinitialization of the transformation matrix: ```void deInitKaleidoscope(KaleidoscopeHandle *handler)```

Alternatively you can directly use the command line program to create kaleidoscope effect with ```./kaleidoscope-cmd <Input Image Path> <Output Image Path> <N>```. You can see an example below for ```N=8```
<div align="center">
    <img src="doc/images/ac-synin.jpg" width="425"/> <img src="doc/images/ac-synin-out.jpg" width="425"/>
    <br>
    <small>Image source: AC Valhalla</small>
</div>

## Benchmark

It is really fast! On a Intel i7-11800H CPU it achieves ~135 FPS for a Full HD (1920 x 1080) image on single thread

If you want to benchmark code on your system make sure you configured with ```-DCMAKE_BUILD_TYPE=Release``` and use this command,

```./kaleidoscope-cmd <Input Image Path> <Output Image Path> <N> <Number of loop>```
