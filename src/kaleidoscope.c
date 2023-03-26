#include "kaleidoscope.h"
#include "kaleidoscope-config.h"

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GCOVR_EXCL_START
void getKaleidoscopeVersion(int *major, int *minor, int *patch)
{
	if (major && minor && patch)
	{
		*major = PROJECT_MAJOR_VERSION;
		*minor = PROJECT_MINOR_VERSION;
		*patch = PROJECT_PATCH_VERSION;
	}
}

char *getKaleidoscopeVersionString()
{
	static char info[sizeof(PROJECT_VERSION)];
	strncpy(info, PROJECT_VERSION, sizeof(PROJECT_VERSION));
	return info;
}

char *getKaleidoscopeLibraryInfo()
{
	int offset = 0;
	static char info[125];

	strncpy(info, PROJECT_VERSION, sizeof(PROJECT_VERSION));
	offset += sizeof(PROJECT_VERSION);
	memset(&info[offset - 1], 32, 1);
	strncpy(&info[offset], COMPILER_NAME, sizeof(COMPILER_NAME));
	offset += sizeof(COMPILER_NAME);
	memset(&info[offset - 1], 32, 1);
	strncpy(&info[offset], COMPILER_VERSION, sizeof(COMPILER_VERSION));
	offset += sizeof(COMPILER_VERSION);
	memset(&info[offset - 1], 32, 1);
	strncpy(&info[offset], BUILD_TYPE, sizeof(BUILD_TYPE));
	offset += sizeof(BUILD_TYPE);
	memset(&info[offset - 1], 32, 1);
	strncpy(&info[offset], PROJECT_BUILD_DATE, sizeof(PROJECT_BUILD_DATE));
	offset += sizeof(PROJECT_BUILD_DATE);
	memset(&info[offset - 1], 32, 1);
	strncpy(&info[offset], PROJECT_BUILD_TIME, sizeof(PROJECT_BUILD_TIME));
	offset += sizeof(PROJECT_BUILD_TIME);
	memset(&info[offset - 1], 32, 1);
#ifdef KALEIDOSCOPE_ENABLE_CUDA
	strncpy(&info[offset], "with CUDA", sizeof("with CUDA"));
	offset += sizeof("with CUDA");
	memset(&info[offset - 1], 32, 1);
	strncpy(&info[offset], CUDA_COMPILER_VERSION, sizeof(CUDA_COMPILER_VERSION));
	offset += sizeof(CUDA_COMPILER_VERSION);
	memset(&info[offset - 1], 32, 1);
#else
	strncpy(&info[offset], "without CUDA", sizeof("without CUDA"));
	offset += sizeof("without CUDA");
	memset(&info[offset - 1], 32, 1);
#endif

	return info;
}
// GCOVR_EXCL_STOP

int compare(const void *lhsPtr, const void *rhsPtr)
{
	struct TransformationInfo_t *lhs = (struct TransformationInfo_t *)lhsPtr;
	struct TransformationInfo_t *rhs = (struct TransformationInfo_t *)rhsPtr;
	return lhs->dstOffset - rhs->dstOffset;
}

void interpolate(struct TransformationInfo_t *dataOut, struct TransformationInfo_t *dataIn, int width, int height)
{
	int idx, jdx;

	// Very simple implementation of nearest neighbour interpolation
	for (idx = 1; idx < height - 1; ++idx)
	{
		int heightOffset = idx * width;
		for (jdx = 1; jdx < width - 1; ++jdx)
		{
			struct TransformationInfo_t *ptrIn = &dataIn[heightOffset + jdx];
			struct TransformationInfo_t *ptrOut = &dataOut[heightOffset + jdx];
			if (!(ptrIn->dstLocation.x) && !(ptrIn->dstLocation.y))
			{
				ptrOut->dstLocation.x = jdx;
				ptrOut->dstLocation.y = idx;
				if (((ptrIn - 1)->dstLocation.x) || ((ptrIn - 1)->dstLocation.y)) // Left
				{
					ptrOut->srcLocation.x = (ptrIn - 1)->srcLocation.x + 1;
					ptrOut->srcLocation.y = (ptrIn - 1)->srcLocation.y;
				}
				else if (((ptrIn + 1)->dstLocation.x) || ((ptrIn + 1)->dstLocation.y)) // Right
				{
					ptrOut->srcLocation.x = (ptrIn + 1)->srcLocation.x - 1;
					ptrOut->srcLocation.y = (ptrIn + 1)->srcLocation.y;
				}
				else if (((ptrIn - width)->dstLocation.x) || ((ptrIn - width)->dstLocation.y)) // Top
				{
					ptrOut->srcLocation.x = (ptrIn - width)->srcLocation.x;
					ptrOut->srcLocation.y = (ptrIn - width)->srcLocation.y - 1;
				}
				else if (((ptrIn + width)->dstLocation.x) || ((ptrIn + width)->dstLocation.y)) // Bottom
				{
					ptrOut->srcLocation.x = (ptrIn + width)->srcLocation.x;
					ptrOut->srcLocation.y = (ptrIn + width)->srcLocation.y + 1;
				}
				else if (((ptrIn - width - 1)->dstLocation.x) || ((ptrIn - width - 1)->dstLocation.y)) // Top-Left
				{
					ptrOut->srcLocation.x = (ptrIn - width - 1)->srcLocation.x - 1;
					ptrOut->srcLocation.y = (ptrIn - width - 1)->srcLocation.y - 1;
				}
				else if (((ptrIn - width + 1)->dstLocation.x) || ((ptrIn - width + 1)->dstLocation.y)) // Top-Right
				{
					ptrOut->srcLocation.x = (ptrIn - width + 1)->srcLocation.x + 1;
					ptrOut->srcLocation.y = (ptrIn - width + 1)->srcLocation.y - 1;
				}
				else if (((ptrIn + width - 1)->dstLocation.x) || ((ptrIn + width - 1)->dstLocation.y)) // Bottom-Left
				{
					ptrOut->srcLocation.x = (ptrIn + width - 1)->srcLocation.x - 1;
					ptrOut->srcLocation.y = (ptrIn + width - 1)->srcLocation.y - 1;
				}
				else if (((ptrIn + width + 1)->dstLocation.x) || ((ptrIn + width + 1)->dstLocation.y)) // Bottom-Right
				{
					ptrOut->srcLocation.x = (ptrIn + width + 1)->srcLocation.x + 1;
					ptrOut->srcLocation.y = (ptrIn + width + 1)->srcLocation.y + 1;
				}
				else
					memset(ptrOut, 0, sizeof(struct TransformationInfo_t));
			}
			else
				*ptrOut = *ptrIn;
		}
	}
}

void rotatePoints(struct TransformationInfo_t *outData, struct TransformationInfo_t *orgData, int width, int height, double angle)
{
	int idx;
	double cosVal = cos(angle * M_PI / 180);
	double sinVal = sin(angle * M_PI / 180);

	for (idx = 0; idx < width * height; ++idx)
	{
		if (orgData[idx].dstLocation.x || orgData[idx].dstLocation.y)
		{
			int newX = (int)round(orgData[idx].dstLocation.x * cosVal + orgData[idx].dstLocation.y * sinVal);
			int newY = (int)round(orgData[idx].dstLocation.y * cosVal - orgData[idx].dstLocation.x * sinVal);

			// Fix origin to top left again
			newX += (width / 2);
			newY += (height / 2);

			if (newX <= width && newX >= 0 && newY <= height && newY >= 0)
			{
				outData[newY * width + newX].srcLocation = orgData[idx].srcLocation;
				outData[newY * width + newX].dstLocation.x = newX;
				outData[newY * width + newX].dstLocation.y = newY;
			}
		}
	}
}

int sliceTriangle(struct TransformationInfo_t *transformPtr, int width, int height, int n, double scaleDown)
{
	int idx, jdx;

	// Variables
	const double topAngle = 360.0 / n;
	const double tanVal = tan(topAngle / 2.0 * M_PI / 180.0); // tan(topAngle / 2) in radians
	const int triangleHeight = (int)fmin(round(width / (2.0 * tanVal)), height - 1);
	const int heightStart = (height - triangleHeight) / 2;
	const int heightEnd = (height + triangleHeight) / 2;
	const int scaleDownOffset = (int)(height * scaleDown / 2);

	// Ensure limits within image
	assert(heightStart >= 0);
	assert(heightStart <= height);
	assert(heightEnd >= 0);
	assert(heightEnd <= height);

	for (idx = heightStart; idx < heightEnd; ++idx)
	{
		const int currentBaseLength = (int)((idx - heightStart) * tanVal);

		const int widthStart = (width / 2 - currentBaseLength);
		const int widthEnd = (width / 2 + currentBaseLength);

		// Ensure limits within image
		if (widthStart < 0 || widthStart > width || widthEnd < 0 || widthEnd > width)
			continue;

		struct TransformationInfo_t *ptr = &transformPtr[idx * width];
		for (jdx = widthStart; jdx <= widthEnd; ++jdx)
		{
			ptr[jdx].srcLocation.x = jdx;
			ptr[jdx].srcLocation.y = idx;

			// Calculate coordinates respect to center to prepare rotating
			ptr[jdx].dstLocation.x = (int)((jdx - width / 2) * scaleDown);
			ptr[jdx].dstLocation.y = (int)((idx - heightStart - height / 2) * scaleDown + scaleDownOffset);
		}
	}

	return EXIT_SUCCESS;
}

int initKaleidoscope(struct KaleidoscopeHandler_t *handler, double k, int n, int width, int height, int nComponents,
					 double scaleDown)
{
	int idx, jdx;

	int retval = EXIT_FAILURE;
	const int nPixels = width * height;
	struct TransformationInfo_t *buffPtr1 = NULL, *buffPtr2 = NULL;

	// Check parameters
	assert(handler);
	assert(n > 2);
	assert(width > 0);
	assert(height > 0);
	assert(nComponents > 0);
	assert(scaleDown > 0.0);
	assert(scaleDown < 1.0);
	assert(k >= 0.0);
	assert(k <= 1.0);

	handler->width = width;
	handler->height = height;
	handler->nComponents = nComponents;
	handler->k = k;

	buffPtr1 = (struct TransformationInfo_t *)calloc(nPixels, sizeof(struct TransformationInfo_t));
	buffPtr2 = (struct TransformationInfo_t *)calloc(nPixels, sizeof(struct TransformationInfo_t));
	if (!buffPtr1 || !buffPtr2)
		goto cleanup;

	if (sliceTriangle(buffPtr1, width, height, n, scaleDown))
		goto cleanup;

	// Rotate all points and fix origin to left top
	for (idx = 0; idx < n; ++idx)
	{
		double rotationAngle = idx * (360.0 / n);
		rotatePoints(buffPtr2, buffPtr1, width, height, rotationAngle);
	}

	// Fill rotation artifacts
	memset(buffPtr1, 0, sizeof(struct TransformationInfo_t) * width * height);
	interpolate(buffPtr1, buffPtr2, width, height);

	// Remove zeros and set to points for handler
	handler->nPoints = 0;
	for (idx = 0; idx < nPixels; ++idx)
	{
		struct TransformationInfo_t *ptr = &buffPtr1[idx];
		if (!(ptr->srcLocation.x) || !(ptr->srcLocation.y))
			continue;

		buffPtr1[handler->nPoints] = *ptr;
		buffPtr1[handler->nPoints].srcOffset =
			ptr->srcLocation.x * nComponents + ptr->srcLocation.y * width * nComponents;
		buffPtr1[handler->nPoints].dstOffset =
			ptr->dstLocation.x * nComponents + ptr->dstLocation.y * width * nComponents;
		++(handler->nPoints);
	}

	// Sort
	qsort(buffPtr1, handler->nPoints, sizeof(struct TransformationInfo_t), compare);

	// Deduplicate
	jdx = 0;
	for (idx = 1; idx < handler->nPoints; ++idx)
	{
		if (compare(&buffPtr1[jdx], &buffPtr1[idx]))
		{
			buffPtr1[jdx] = buffPtr1[idx];
			++jdx;
		}
	}
	handler->nPoints = jdx;

	handler->pTransferFunc = (struct TransformationInfo_t *)malloc(handler->nPoints * sizeof(struct TransformationInfo_t));
	memcpy(handler->pTransferFunc, buffPtr1, handler->nPoints * sizeof(struct TransformationInfo_t));
	retval = EXIT_SUCCESS;

cleanup:
	free(buffPtr1);
	free(buffPtr2);

	if (retval == EXIT_FAILURE)
		free(handler->pTransferFunc);

	return retval;
}

void processKaleidoscope(struct KaleidoscopeHandler_t *handler, unsigned char *imgIn, unsigned char *imgOut)
{
	unsigned long long idx;
	const unsigned long long nPixels = (unsigned long long)handler->width * handler->height * handler->nComponents;

	unsigned char *srcPtr = imgIn;
	unsigned char *destPtr = imgOut;
	struct TransformationInfo_t *ptrTransform = &(handler->pTransferFunc[0]);

	for (idx = 0; idx < nPixels; ++idx, ++destPtr, ++srcPtr) // Dim image
		*destPtr = (unsigned char)((*srcPtr) * handler->k);
	for (idx = 0; idx < handler->nPoints; ++idx, ++ptrTransform) // Merge
		memcpy(&(imgOut[ptrTransform->dstOffset]), &(imgIn[ptrTransform->srcOffset]), handler->nComponents);
}

void deInitKaleidoscope(struct KaleidoscopeHandler_t *handler)
{
	if (handler)
		free(handler->pTransferFunc);
}
