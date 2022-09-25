#ifndef _JPEG_UTILS_H_
#define _JPEG_UTILS_H_

#include <inttypes.h>
#include <stdlib.h>

/**
 * @brief Data struct for images
 */
struct ImageData_t
{
	int width;
	int height;
	unsigned char nComponent;
	char format;
	unsigned char *data;
};
typedef struct ImageData_t ImageData;

/**
 * @brief Get image data from an input file
 *
 * @param[in] path Path to input image file
 * @param[out] img Image data
 * @return int  Returns SUCCESS or FAIL
 */
int readImage(const char *path, ImageData *img);

/**
 * @brief Save image data to an output file
 *
 * @param[in] path Path to output image file
 * @param[in] img Image data
 * @param[in] jpegQuality Quality of the output image
 * @return int Returns SUCCESS or FAIL
 */
int saveImage(const char *path, ImageData *img, int jpegQuality);

/**
 * @brief Free memory allocated by read image
 * @param[in] img Image data
 */
static inline void freeImageData(ImageData *img) { free(img->data); }

#endif // _JPEG_UTILS_H_