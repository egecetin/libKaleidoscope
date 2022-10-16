<div align="center">

# Kaleidoscope Library
A library to create kaleidoscope effect on images. You can build on all platforms using CMake.

 ![GitHub](https://img.shields.io/badge/Language-C-informational?style=plastic)
 ![GitHub](https://img.shields.io/github/license/egecetin/kaleidoscope?style=plastic)
 ![GitHub last commit](https://img.shields.io/github/last-commit/egecetin/kaleidoscope?style=plastic)
</div>

## Building

Use the following commands,

```
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage

The library have a simple usage you only need three functions to use. Check the sample usage at ```src/kaleidoscope-cmd.c```

- Initialization of the transformation matrix: ```int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, double scaleDown)```
- Processing image (Can be used multiple times if the input image has same dimensions): ```void processKaleidoscope(KaleidoscopeHandle *handler, double k, ImageData *imgIn, ImageData *imgOut)```
- Deinitialization of the transformation matrix: ```void deInitKaleidoscope(KaleidoscopeHandle *handler)```

Alternatively you can directly use the command line program to create kaleidoscope effect!

```./kaleidoscope <Input Image Path> <Output Image Path> <N> ```

You can see an example below for ```N=8```

<div align="center">
<img src="data/ac-synin.jpg" width="425"/> <img src="data/ac-synin-out.jpg" width="425"/>
<br>
<small>Image source: AC Valhalla</small>
</div>
