#ifndef _KALEIDOSCOPE_H_
#define _KALEIDOSCOPE_H_

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
	/// Location from source image
	Point2D srcLocation;
	/// Location to destination image
	Point2D dstLocation;
	/// Offset from source image
	unsigned long long srcOffset;
	/// Offset from destination image
	unsigned long long dstOffset;
};
typedef struct TransformationInfo_t TransformationInfo;

/**
 * @brief Struct for kaleidoscope effect generator
 */
struct KaleidoscopeHandle_t
{
	/// Image width
	int width;
	/// Image height
	int height;
	/// Number of components (eg 3 for RGB)
	unsigned char nComponents;
	/// Total number of points of transfer function
	long long nPoints;
	/// Transformation info
	struct TransformationInfo_t *pTransferFunc;
};
typedef struct KaleidoscopeHandle_t KaleidoscopeHandle;

/**
 * @brief Get the Kaleidoscope Library version as integer
 * @param[in, out] major Major number
 * @param[in, out] minor Minor number
 * @param[in, out] patch Patch number
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
void interpolate(TransformationInfo *dataOut, const TransformationInfo *dataIn, int width, int height);

/**
 * @brief Rotates the coordinates of sliced triangle. Internal use only
 * @param[out] outData Rotated data
 * @param[in] orgData Sliced data coordinates
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 * @param[in] angle Top angle of sliced triangle
 */
void rotatePoints(TransformationInfo *outData, const TransformationInfo *orgData, int width, int height, double angle);

/**
 * @brief Slices a suitable triangle from image
 * @param[out] transformPtr Sliced triangle coordinates
 * @param[in] width Width of input image
 * @param[in] height Height of input image
 * @param[in] n Number of images for effect
 * @param[in] scaleDown Scale down ratio to shrink image
 */
void sliceTriangle(TransformationInfo *transformPtr, int width, int height, int n, double scaleDown);

/**
 * @brief Initializes kaleidoscope handler
 * @param[in, out] handler Kaleidoscope effect handler
 * @param[in] n Number of images for effect
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] nComponents Number of image components (eg 3 for RGB)
 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
 * @return int 0 on success, negative otherwise
 */
int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, int nComponents, double scaleDown);

/**
 * @brief Applies kaleidoscope effect to image
 * @param[in] handler Kaleidoscope effect handler
 * @param[in] k Variable to dim background. Should be between 0.0 and 1.0
 * @param[in] imgIn Input image
 * @param[out] imgOut Output image
 */
void processKaleidoscope(const KaleidoscopeHandle *handler, double k, const unsigned char *imgIn, unsigned char *imgOut);

/**
 * @brief Deinitializes kaleidoscope handler
 * @param[in] handler Kaleidoscope effect handler
 */
void deInitKaleidoscope(KaleidoscopeHandle *handler);

#endif // _KALEIDOSCOPE_H_
