#pragma once

#include <inttypes.h>

/// Assume always 3 component image
#define COLOR_COMPONENTS 3
/// Quality of the output image
#define JPEG_QUALITY 90

#ifndef nullptr
#define nullptr NULL
#endif

#define SUCCESS 0;
#define FAIL -1;

/**
 * @brief Data struct for pixels
 */
struct PointData_t
{
	int32_t x;
	int32_t y;
	uint8_t value[COLOR_COMPONENTS];
};
typedef struct PointData_t PointData;

/**
 * @brief Data struct for images
 */
struct ImageData_t
{
	uint32_t width;
	uint32_t height;
	uint8_t *data;
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
 * @return int Returns SUCCESS or FAIL
 */
int saveImage(const char *path, ImageData *img);

/**
 * @brief Dim the whole image
 *
 * @param[in, out] img Input image data
 * @param[in] k Background dimming scale
 * @param[out] out (Optional) Output image data. If it is null operation will be in-place
 * @return int Returns SUCCESS or FAIL
 */
static inline int dimBackground(ImageData *img, double k, ImageData *out);

/**
 * @brief Slice a scaled triangle from image
 *
 * @param[in] img Input image
 * @param[out] slicedData Output sliced image data
 * @param[out] len Length of the output data
 * @param[in] n N for kaleidoscope effect
 * @param[in] scaleDown Scale factor of sliced data (Should be less than 0.5)
 * @return int Returns SUCCESS or FAIL
 */
static inline int sliceTriangle(ImageData *img, PointData **slicedData, uint64_t *len, int n, double scaleDown);

/**
 * @brief Merge sliced img to main image 
 *
 * @param[in, out] img Source image
 * @param[in] slicedData Sliced triangle
 * @param[out] hitData Data points for interpolation
 * @param[in] len Length of the slicedData
 * @param[in] n N for kaleidoscope effect
 * @return int Returns SUCCESS or FAIL
 */
static inline int rotateAndMerge(ImageData *img, PointData *slicedData, uint64_t len, uint8_t *hitData, int n);

/**
 * @brief Very simple implementation of nearest neighbour interpolation
 *
 * @param[in, out] img Merged image
 * @param[in] hitData Modified point info
 * @return int
 */
static inline int interpolate(ImageData *img, uint8_t *hitData);

/**
 * @brief Main function
 *
 * @param[in, out] img Input image
 * @param[in] n N for kaleidoscope effect
 * @param[in] k Background dimming scale
 * @param[in] scaleDown Scale factor for sliced image
 * @return int Returns SUCCESS or FAIL
 */
int kaleidoscope(ImageData *img, int n, double k, double scaleDown);
