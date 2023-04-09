#ifndef _KALEIDOSCOPE_H_
#define _KALEIDOSCOPE_H_

#include <c/kaleidoscope-definitions.h>

/**
 * @brief Get the Kaleidoscope Library version as integer
 * @param[out] major Major number
 * @param[out] minor Minor number
 * @param[out] patch Patch number
 */
void getKaleidoscopeVersion(int *major, int *minor, int *patch);

/**
 * @brief Get the Kaleidoscope Library version as string
 * @return char* Library version
 */
char *getKaleidoscopeVersionString();

/**
 * @brief Get the Kaleidoscope Library info as string
 * @return char* Library information
 */
char *getKaleidoscopeLibraryInfo();

/**
 * @brief A simple interpolation function. Internal use only
 * @param[out] dataOut Output (interpolated) binary image
 * @param[in] dataIn Input binary image
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 */
void interpolate(struct TransformationInfo_t *dataOut, struct TransformationInfo_t *dataIn, int width, int height);

/**
 * @brief Rotates the coordinates of sliced triangle. Internal use only
 * @param[out] outData Rotated data
 * @param[in] orgData Sliced data coordinates
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 * @param[in] angle Top angle of sliced triangle
 */
void rotatePoints(struct TransformationInfo_t *outData, struct TransformationInfo_t *orgData, int width, int height, double angle);

/**
 * @brief Slices a suitable triangle from image
 * @param[out] transformPtr Sliced triangle coordinates
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 * @param[in] n Number of images for effect
 * @param[in] scaleDown Scale down ratio to shrink image
 * @return int 0 on success, -1 otherwise
 */
int sliceTriangle(struct TransformationInfo_t *transformPtr, int width, int height, int n, double scaleDown);

/**
 * @brief Initializes kaleidoscope handler
 * @param[in, out] handler Kaleidoscope effect handler
 * @param[in] k Variable to dim background. Should be between 0.0 and 1.0
 * @param[in] n Number of images for effect
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] nComponents Number of image components (eg 3 for RGB)
 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
 * @return int 0 on success, negative otherwise
 */
int initKaleidoscope(struct KaleidoscopeHandler_t *handler, double k, int n, int width, int height, int nComponents,
					 double scaleDown);

/**
 * @brief Applies kaleidoscope effect to image
 * @param[in] handler Kaleidoscope effect handler
 * @param[in] imgIn Input image
 * @param[out] imgOut Output image
 */
void processKaleidoscope(struct KaleidoscopeHandler_t *handler, unsigned char *imgIn, unsigned char *imgOut);

/**
 * @brief Deinitializes kaleidoscope handler
 * @param[in] handler Kaleidoscope effect handler
 */
void deInitKaleidoscope(struct KaleidoscopeHandler_t *handler);

#endif // _KALEIDOSCOPE_H_
