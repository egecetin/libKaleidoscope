#ifndef _JPEG_UTILS_H_
#define _JPEG_UTILS_H_

#include "kaleidoscope.h"

#include <turbojpeg.h>

/**
 * @brief Get image data from an input file
 * @param[in] path Path to input image file
 * @param[out] img Image data
 * @return int Returns 0 on success
 */
int readImage(const char *path, ImageData *img);

/**
 * @brief Save image data to an output file
 * @param[in] path Path to output image file
 * @param[in] img Image data
 * @param[in] pixelFormat Pixel format
 * @param[in] samplingFormat Sampling format
 * @param[in] jpegQuality Quality of the output image
 * @return int Returns 0 on success
 */
int saveImage(const char *path, ImageData *img, enum TJPF pixelFormat, enum TJSAMP samplingFormat, int jpegQuality);

#endif // _JPEG_UTILS_H_
