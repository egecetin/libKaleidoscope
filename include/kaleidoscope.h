#ifndef _KALEIDOSCOPE_H_
#define _KALEIDOSCOPE_H_

/**
 * @brief Data struct for images
 */
struct ImageData_t
{
	int width;
	int height;
	unsigned char nComponents;
	unsigned char *data;
};
typedef struct ImageData_t ImageData;

/**
 * @brief Data struct for pixel locations
 */
struct Point2D_t
{
	int x;
	int y;
};
typedef struct Point2D_t Point2D;

/**
 * @brief Data struct for transformation information
 */
struct TransformationInfo_t
{
	// Location from source image
	Point2D srcLocation;
	// Location to destination image
	Point2D dstLocation;
};
typedef struct TransformationInfo_t TransformationInfo;

/**
 * @brief Struct for kaleidoscope effect generator
 */
struct KaleidoscopeHandle_t
{
	unsigned long long nPoints;
	TransformationInfo *pTransferFunc;
};
typedef struct KaleidoscopeHandle_t KaleidoscopeHandle;

/**
 * @brief A simple interpolation function. Internal use only
 * @param[out] dataOut Output (interpolated) binary image
 * @param[in] dataIn Input binary image
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 */
void interpolate(TransformationInfo *dataOut, TransformationInfo *dataIn, int width, int height);

/**
 * @brief Rotates the coordinates of sliced triangle. Internal use only
 * @param[out] outData Rotated data
 * @param[in] orgData Sliced data coordinates
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 * @param[in] angle Top angle of sliced triangle
 */
void rotatePoints(TransformationInfo *outData, TransformationInfo *orgData, int width, int height, double angle);

/**
 * @brief Slices a suitable triangle from image
 * @param[out] transformPtr Sliced triangle coordinates
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 * @param[in] n Number of images for effect
 * @param[in] scaleDown Scale down ratio to shrink image
 * @return int 0 on success, -1 otherwise
 */
int sliceTriangle(TransformationInfo *transformPtr, int width, int height, int n, double scaleDown);

/**
 * @brief Initializes kaleidoscope handler
 * @param[in, out] handler Kaleidoscope effect handler
 * @param[in] n Number of images for effect
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
 * @return int 0 on success, negative otherwise
 */
int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, double scaleDown);

/**
 * @brief Applies kaleidoscope effect to image
 * @param[in] handler Kaleidoscope effect handler
 * @param[in] k Variable to dim background. Should be between 0.0 and 1.0
 * @param[in] imgIn Input image
 * @param[out] imgOut Output image
 */
void processKaleidoscope(KaleidoscopeHandle *handler, double k, ImageData *imgIn, ImageData *imgOut);

/**
 * @brief Deinitializes kaleidoscope handler
 * @param[in] handler Kaleidoscope effect handler
 */
void deInitKaleidoscope(KaleidoscopeHandle *handler);

/**
 * @brief Allocates memory for image
 * @param[in] img Image data
 * @param[in] width Width of image
 * @param[in] height Height of image
 * @param[in] nComponents Number of components
 * @return int Returns 0 on success
 */
int initImageData(ImageData *img, int width, int height, int nComponents);

/**
 * @brief Free memory allocated by read image
 * @param[in] img Image data
 */
void deInitImageData(ImageData *img);

#endif // _KALEIDOSCOPE_H_
