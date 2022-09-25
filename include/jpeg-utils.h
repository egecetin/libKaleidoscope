#ifndef _JPEG_UTILS_H_
#define _JPEG_UTILS_H_

#include "kaleidoscope.h"

#include <inttypes.h>
#include <stdlib.h>

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

/**
 * @brief Allocates memory for image
 * @param[in] img Image data
 * @param[in] width Width of image
 * @param[in] height Height of image
 * @param[in] nComponents Number of components
 * @return int Returns 0 on success
 */
static inline int initImageData(ImageData *img, int width, int height, int nComponents);

/**
 * @brief Free memory allocated by read image
 * @param[in] img Image data
 */
static inline void freeImageData(ImageData *img) { free(img->data); }

#endif // _JPEG_UTILS_H_