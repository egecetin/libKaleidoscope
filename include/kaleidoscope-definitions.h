#ifndef _KALEIDOSCOPE_DEFINITIONS_H_
#define _KALEIDOSCOPE_DEFINITIONS_H_

#include "kaleidoscope-config.h"

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
	TransformationInfo *pTransferFunc;
	/// Calculated block size for dim operation on GPU
	int blockSizeDim;
	/// Calculated grid size for dim operation on GPU
	int gridSizeDim;
	/// Calculated block size for transform itself on GPU
	int blockSizeTransform;
	/// Calculated grid size for transform itself on GPU
	int gridSizeTransform;
};
typedef struct KaleidoscopeHandle_t KaleidoscopeHandle;

#endif // _KALEIDOSCOPE_DEFINITIONS_H_
