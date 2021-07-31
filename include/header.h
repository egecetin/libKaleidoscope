#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <turbojpeg.h>

/// Assume always 3 component image
#define COLOR_COMPONENTS 3
/// Quality of the output image
#define JPEG_QUALITY 90

#ifndef nullptr
#define nullptr NULL
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

/**
 * @brief       Get image data from an input file
 * 
 * @param path  Path to input image file
 * @param img   Image data
 * @return int  Returns SUCCESS or FAIL
 */
int readImage(const char *path, ImageData *img);

/**
 * @brief       Save image data to an output file
 * 
 * @param path  Path to output image file
 * @param img   Image data
 * @return int  Returns SUCCESS or FAIL
 */
int saveImage(const char *path, ImageData *img);

/**
 * @brief       Dim the whole image
 * 
 * @param img   Input image data
 * @param k     Gamma correction rate
 * @param out   (Optional) Output image data. If it is null operation will be in-place
 * @return int  Returns SUCCESS or FAIL
 */
int dimBackground(ImageData *img, float k, ImageData *out);

/**
 * @brief 
 * 
 * @param img 
 * @param k 
 * @return int 
 */
int kaleidoscope(ImageData *img, float k);
