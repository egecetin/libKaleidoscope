#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
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

struct PointData_t
{
    int32_t x;
    int32_t y;
    uint8_t value[COLOR_COMPONENTS];
};
typedef struct PointData_t PointData;

struct ImageData_t
{
    uint32_t width;
    uint32_t height;
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
 * @param k     Background dimming scale
 * @param out   (Optional) Output image data. If it is null operation will be in-place
 * @return int  Returns SUCCESS or FAIL
 */
int dimBackground(ImageData *img, float k, ImageData *out);

/**
 * @brief               Slice a scaled triangle from image
 * 
 * @param img           Input image
 * @param slicedData    Output sliced image data
 * @param len           Length of the output data
 * @param n             N for kaleidoscope effect
 * @param scaleDown     Scale factor of sliced data (Should be less than 0.5)
 * @return int          Returns SUCCESS or FAIL
 */
int sliceTriangle(ImageData *img, PointData **slicedData, uint64_t *len, int n, float scaleDown);

/**
 * @brief           Main function
 * 
 * @param img       Input image
 * @param n         N for kaleidoscope effect
 * @param k         Background dimming scale
 * @param scaleDown Scale factor for sliced image
 * @return int      Returns SUCCESS or FAIL
 */
int kaleidoscope(ImageData *img, int n, float k, float scaleDown);
