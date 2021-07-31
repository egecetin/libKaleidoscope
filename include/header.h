#pragma once

#include <stdio.h>
#include <inttypes.h>

#include <turbojpeg.h>

/// Assume always colored image
#define COLOR_COMPONENTS 3
/// Quality of the output image
#define JPEG_QUALITY 90

#ifndef nullptr
#define nullptr NULL;
#endif

#define SUCCESS 0;
#define FAIL -1;

struct ImageData_t
{
    uint32_t width;
    uint32_t height;
    uint64_t size;
    uint8_t *data;
};
typedef struct ImageData_t ImageData;

int readImage(char *path, ImageData *img);
int saveImage(char *path, ImageData *img);

int dimBackground();