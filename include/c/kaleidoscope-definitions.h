#ifndef _KALEIDOSCOPE_DEFINITIONS_H_
#define _KALEIDOSCOPE_DEFINITIONS_H_

/**
 * @brief Data struct for pixel locations
 */
struct Point2D_t
{
	int x;
	int y;
};

/**
 * @brief Data struct for transformation information
 */
struct TransformationInfo_t
{
	/// Location from source image
	struct Point2D_t srcLocation;
	/// Location to destination image
	struct Point2D_t dstLocation;
	/// Offset from source image
	unsigned long long srcOffset;
	/// Offset from destination image
	unsigned long long dstOffset;
};

/**
 * @brief Struct for kaleidoscope effect generator
 */
struct KaleidoscopeHandler_t
{
	/// Dim constant
	double k;
	/// Image width
	int width;
	/// Image height
	int height;
	/// Number of components (eg 3 for RGB)
	unsigned char nComponents;
	/// Total number of points of transfer function
	unsigned long long nPoints;
	/// Transformation info
	struct TransformationInfo_t *pTransferFunc;
};

#endif // _KALEIDOSCOPE_DEFINITIONS_H_
