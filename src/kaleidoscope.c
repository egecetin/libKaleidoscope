#include "kaleidoscope.h"

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void interpolate(TransformationInfo *dataIn, TransformationInfo *dataOut, int width, int height)
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

int initKaleidoscope(KaleidoscopeHandle *handler, int n, int width, int height, double scaleDown)
{
	int idx, jdx;
	unsigned long long ctr;
	int retval = EXIT_FAILURE;

	// Parameters of triangle
	const double topAngle = 360.0 / n;
	const double tanVal = tan(topAngle / 2.0 * M_PI / 180.0); // tan(topAngle / 2) in radians
	const int triangleHeight = (int)fmin(round(width / (2.0 * tanVal)), height - 1);

	// Offsets
	const int heightStart = (height - triangleHeight) / 2;
	const int heightEnd = (height + triangleHeight) / 2;
	const int scaleDownOffset = (int)(height * scaleDown / 2);

	// Total number of pixels
	const int nPixels = width * height;

	// Init same size arrays for simplicity to determine target pixel coordinates
	TransformationInfo *buffPtr1 = NULL, *buffPtr2 = NULL;

	// Check parameters
	if (!handler || n <= 2 || width <= 0 || height <= 0 || scaleDown < 0.0 || scaleDown > 1.0)
		return retval;

	buffPtr1 = (TransformationInfo *)calloc(nPixels, sizeof(TransformationInfo));
	buffPtr2 = (TransformationInfo *)calloc(nPixels, sizeof(TransformationInfo));
	if (!buffPtr1 || !buffPtr2)
		goto cleanup;

	// Ensure limits within image
	if (heightStart < 0 || heightStart > height || heightEnd < 0 || heightEnd > height)
		goto cleanup;

	for (idx = heightStart; idx < heightEnd; ++idx)
	{
		const int currentBaseLength = (int)((idx - heightStart) * tanVal);

		const int widthStart = (width / 2 - currentBaseLength);
		const int widthEnd = (width / 2 + currentBaseLength);

		// Ensure limits within image
		if (widthStart < 0 || widthStart > width || widthEnd < 0 || widthEnd > width)
			continue;

		TransformationInfo *ptr = &buffPtr1[idx * width];
		for (jdx = widthStart; jdx <= widthEnd; ++jdx)
		{
			ptr[jdx].srcLocation.x = jdx;
			ptr[jdx].srcLocation.y = idx;

			// Calculate coordinates respect to center to prepare rotating
			ptr[jdx].dstLocation.x = (int)((jdx - width / 2) * scaleDown);
			ptr[jdx].dstLocation.y = (int)((idx - heightStart - height / 2) * scaleDown + scaleDownOffset);
		}
	}

	// Rotate all points and fix origin to left top
	for (idx = 0; idx < n; ++idx)
	{
		double rotationAngle = idx * (360.0 / n);
		rotatePoints(buffPtr2, buffPtr1, width, height, rotationAngle);
	}

	// Fill rotation artifacts
	interpolate(buffPtr2, buffPtr1, width, height);

	// Reduction and set to points for handler
	handler->nPoints = 0;
	for (ctr = 0; ctr < nPixels; ++ctr)
	{
		TransformationInfo *ptr = &buffPtr1[ctr];
		if (!(ptr->srcLocation.x) || !(ptr->srcLocation.y))
			continue;

		buffPtr1[handler->nPoints] = *ptr;
		++(handler->nPoints);
	}

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

void processKaleidoscope(KaleidoscopeHandle *handler, double k, ImageData *imgIn, ImageData *imgOut)
{
	unsigned long long idx;

	// Dim image
	for (idx = 0; idx < imgIn->width * imgIn->height * imgIn->nComponents; ++idx)
		imgOut->data[idx] = (unsigned char)(imgIn->data[idx] * k);
	for (idx = 0; idx < handler->nPoints; ++idx)
	{
		unsigned long long srcIdx = handler->pTransferFunc[idx].srcLocation.y * imgIn->width * imgIn->nComponents +
									handler->pTransferFunc[idx].srcLocation.x * imgIn->nComponents;
		unsigned long long dstIdx = handler->pTransferFunc[idx].dstLocation.y * imgIn->width * imgIn->nComponents +
									handler->pTransferFunc[idx].dstLocation.x * imgIn->nComponents;
		memcpy(&(imgOut->data[dstIdx]), &(imgIn->data[srcIdx]), imgIn->nComponents);
	}
}

void deInitKaleidoscope(KaleidoscopeHandle *handler)
{
	if (handler)
		free(handler->pTransferFunc);
}

int initImageData(ImageData *img, int width, int height, int nComponents)
{
	img->data = (unsigned char *)malloc(width * height * nComponents);
	if (!img->data)
		return EXIT_FAILURE;

	img->height = height;
	img->nComponents = nComponents;
	img->width = width;
	return EXIT_SUCCESS;
}

void deInitImageData(ImageData *img)
{
	if (img)
		free(img->data);
}
