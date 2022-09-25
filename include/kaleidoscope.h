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
	// Length for bulk replacement
	unsigned int length;
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

#endif // _KALEIDOSCOPE_H_