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
	// offset += sizeof(PROJECT_BUILD_TIME);

	return info;
}

static int compare(const void *lhsPtr, const void *rhsPtr)
{
	const TransformationInfo *lhs = (const TransformationInfo *)lhsPtr;
	const TransformationInfo *rhs = (const TransformationInfo *)rhsPtr;
	return lhs->dstOffset - rhs->dstOffset;
}

void interpolate(TransformationInfo *dataOut, TransformationInfo *dataIn, int width, int height)
{
	int idx, jdx;

	// Very simple implementation of nearest neighbour interpolation
	for (idx = 1; idx < height - 1; ++idx)
	{
		int heightOffset = idx * width;
		for (jdx = 1; jdx < width - 1; ++jdx)
		{
			TransformationInfo *ptrIn = &dataIn[heightOffset + jdx];
			TransformationInfo *ptrOut = &dataOut[heightOffset + jdx];
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
					memset(ptrOut, 0, sizeof(TransformationInfo));
			}
			else
				*ptrOut = *ptrIn;
		}
	}
}

void rotatePoints(TransformationInfo *outData, TransformationInfo *orgData, int width, int height, double angle)
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

void sliceTriangle(TransformationInfo *transformPtr, int width, int height, int n, double scaleDown)
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

		TransformationInfo *ptr = &transformPtr[idx * width];
		for (jdx = widthStart; jdx <= widthEnd; ++jdx)
		{
			ptr[jdx].srcLocation.x = jdx;
			ptr[jdx].srcLocation.y = idx;

			// Calculate coordinates respect to center to prepare rotating
			ptr[jdx].dstLocation.x = (int)((jdx - width / 2) * scaleDown);
			ptr[jdx].dstLocation.y = (int)((idx - heightStart - height / 2) * scaleDown + scaleDownOffset);
		}
	}
}

int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, int nComponents, double scaleDown)
{
	int idx, jdx;

	int retval = EXIT_FAILURE;
	const int nPixels = width * height;
	TransformationInfo *buffPtr1 = NULL, *buffPtr2 = NULL;

	// Check parameters
	if (handler == NULL || n <= 2 || width <= 0 || height <= 0 || nComponents <= 0 || scaleDown <= 0.0 ||
		scaleDown >= 1.0)
		return EXIT_FAILURE;

	assert(handler);
	assert(n > 2);
	assert(width > 0);
	assert(height > 0);
	assert(nComponents > 0);
	assert(scaleDown > 0.0);
	assert(scaleDown < 1.0);

	handler->width = width;
	handler->height = height;
	handler->nComponents = nComponents;

	buffPtr1 = (TransformationInfo *)calloc(nPixels, sizeof(TransformationInfo));
	buffPtr2 = (TransformationInfo *)calloc(nPixels, sizeof(TransformationInfo));
	if (!buffPtr1 || !buffPtr2)
		goto cleanup;

	sliceTriangle(buffPtr1, width, height, n, scaleDown);

	// Rotate all points and fix origin to left top
	for (idx = 0; idx < n; ++idx)
	{
		double rotationAngle = idx * (360.0 / n);
		rotatePoints(buffPtr2, buffPtr1, width, height, rotationAngle);
	}

	// Fill rotation artifacts
	memset(buffPtr1, 0, sizeof(TransformationInfo) * width * height);
	interpolate(buffPtr1, buffPtr2, width, height);

	// Remove zeros and set to points for handler
	handler->nPoints = 0;
	for (idx = 0; idx < nPixels; ++idx)
	{
		TransformationInfo *ptr = &buffPtr1[idx];
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
	qsort(buffPtr1, handler->nPoints, sizeof(TransformationInfo), compare);

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

	handler->pTransferFunc = (TransformationInfo *)malloc(handler->nPoints * sizeof(TransformationInfo));
	memcpy(handler->pTransferFunc, buffPtr1, handler->nPoints * sizeof(TransformationInfo));
	retval = EXIT_SUCCESS;

cleanup:
	free(buffPtr1);
	free(buffPtr2);

	if (retval == EXIT_FAILURE)
		free(handler->pTransferFunc);

	return retval;
}

void processKaleidoscope(KaleidoscopeHandle *handler, double k, unsigned char *imgIn, unsigned char *imgOut)
{
	long long idx;
	const long long nComponents = handler->nComponents;
	const long long nPixels = (long long)handler->width * handler->height * handler->nComponents;

	unsigned char *srcPtr = imgIn;
	unsigned char *destPtr = imgOut;
	TransformationInfo *ptrTransform = &(handler->pTransferFunc[0]);

	for (idx = 0; idx < nPixels; ++idx, ++destPtr, ++srcPtr) // Dim image
		*destPtr = (unsigned char)((*srcPtr) * k);
	for (idx = 0; idx < handler->nPoints; ++idx, ++ptrTransform) // Merge
		memcpy(&(imgOut[ptrTransform->dstOffset]), &(imgIn[ptrTransform->srcOffset]), nComponents);
}

void deInitKaleidoscope(KaleidoscopeHandle *handler)
{
	if (handler)
		free(handler->pTransferFunc);
}
