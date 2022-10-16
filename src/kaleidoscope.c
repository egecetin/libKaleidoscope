#include "kaleidoscope.h"

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NDEBUG
#include "jpeg-utils.h"
#endif

void interpolate(TransformationInfo *dataIn, TransformationInfo *dataOut, int width, int height)
{
	// Very simple implementation of nearest neighbour interpolation
	for (int idx = 1; idx < height - 1; ++idx)
	{
		int heightOffset = idx * width;
		for (int jdx = 1; jdx < width - 1; ++jdx)
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
	double cosVal = cos(angle * M_PI / 180);
	double sinVal = sin(angle * M_PI / 180);

	for (int idx = 0; idx < width * height; ++idx)
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
	int retval = EXIT_FAILURE;

	// Parameters of triangle
	const double topAngle = 360.0 / n;
	const double tanVal = tan(topAngle / 2.0 * M_PI / 180.0); // tan(topAngle / 2) in radians
	const int triangleHeight = min((int)round(width / (2.0 * tanVal)), height - 1);

	// Offsets
	const int heightStart = (height - triangleHeight) / 2;
	const int heightEnd = (height + triangleHeight) / 2;
	const int scaleDownOffset = (int)(height * scaleDown / 2);

	// Total number of pixels
	const int nPixels = width * height;

	// Init same size arrays for simplicity to determine target pixel coordinates
	TransformationInfo *buffPtr1 = NULL, *buffPtr2 = NULL;

#ifndef NDEBUG
	// Debug variables
	ImageData imgBuffer;
#endif

	// Check parameters
	if (!handler || n <= 0 || width <= 0 || height <= 0 || scaleDown < 0.0 || scaleDown > 1.0)
		return retval;

	buffPtr1 = (TransformationInfo *)calloc(nPixels, sizeof(TransformationInfo));
	buffPtr2 = (TransformationInfo *)calloc(nPixels, sizeof(TransformationInfo));
	if (!buffPtr1 || !buffPtr2)
		goto cleanup;

#ifndef NDEBUG
	imgBuffer.nComponents = 1;
	imgBuffer.width = width;
	imgBuffer.height = height;
	imgBuffer.data = (unsigned char *)calloc(nPixels, sizeof(unsigned char));
#endif

	// Ensure limits within image
	if (heightStart < 0 || heightStart > height || heightEnd < 0 || heightEnd > height)
		goto cleanup;

	for (int idx = heightStart; idx < heightEnd; ++idx)
	{
		const int currentBaseLength = (int)((idx - heightStart) * tanVal);

		const int widthStart = (width / 2 - currentBaseLength);
		const int widthEnd = (width / 2 + currentBaseLength);

		// Ensure limits within image
		if (widthStart < 0 || widthStart > width || widthEnd < 0 || widthEnd > width)
			continue;

		TransformationInfo *ptr = &buffPtr1[idx * width];
		for (int jdx = widthStart; jdx <= widthEnd; ++jdx)
		{
			ptr[jdx].srcLocation.x = jdx;
			ptr[jdx].srcLocation.y = idx;

			// Calculate coordinates respect to center to prepare rotating
			ptr[jdx].dstLocation.x = (int)((jdx - width / 2) * scaleDown);
			ptr[jdx].dstLocation.y = (int)((idx - heightStart - height / 2) * scaleDown + scaleDownOffset);
		}
	}

#ifndef NDEBUG
	memset(imgBuffer.data, 0, nPixels);
	// Save source mask as image
	for (unsigned long long idx = 0; idx < nPixels; ++idx)
	{
		if (buffPtr1[idx].srcLocation.x && buffPtr1[idx].srcLocation.y)
			imgBuffer.data[idx] = 255;
	}
	saveImage("imgSrcMaskPre.jpg", &imgBuffer, TJPF_GRAY, TJSAMP_GRAY, 90);
#endif

	// Rotate all points and fix origin to left top
	for (int idx = 0; idx < n; ++idx)
	{
		double rotationAngle = idx * (360.0 / n);
		rotatePoints(buffPtr2, buffPtr1, width, height, rotationAngle);
	}

#ifndef NDEBUG
	memset(imgBuffer.data, 0, nPixels);
	// Save destination mask as image
	for (unsigned long long idx = 0; idx < nPixels; ++idx)
	{
		if (buffPtr2[idx].dstLocation.x && buffPtr2[idx].dstLocation.y)
			imgBuffer.data[idx] = 255;
	}
	saveImage("imgDstMaskPre.jpg", &imgBuffer, TJPF_GRAY, TJSAMP_GRAY, 90);
#endif

	// Fill rotation artifacts
	interpolate(buffPtr2, buffPtr1, width, height);

	// Reduction and set to points for handler
	handler->nPoints = 0;
	for (unsigned long long idx = 0; idx < nPixels; ++idx)
	{
		TransformationInfo *ptr = &buffPtr1[idx];
		if (!(ptr->srcLocation.x) || !(ptr->srcLocation.y))
			continue;

		buffPtr1[handler->nPoints] = *ptr;
		++(handler->nPoints);
	}

#ifndef NDEBUG
	// Save final source mask as image
	memset(imgBuffer.data, 0, nPixels);
	for (unsigned long long idx = 0; idx < handler->nPoints; ++idx)
	{
		if (buffPtr1[idx].srcLocation.x && buffPtr1[idx].srcLocation.y)
			imgBuffer.data[buffPtr1[idx].srcLocation.y * width + buffPtr1[idx].srcLocation.x] = 255;
	}
	saveImage("imgSrcMaskPost.jpg", &imgBuffer, TJPF_GRAY, TJSAMP_GRAY, 90);

	memset(imgBuffer.data, 0, nPixels);
	for (unsigned long long idx = 0; idx < handler->nPoints; ++idx)
	{
		if (buffPtr1[idx].dstLocation.x && buffPtr1[idx].dstLocation.y)
			imgBuffer.data[buffPtr1[idx].dstLocation.y * width + buffPtr1[idx].dstLocation.x] = 255;
	}
	saveImage("imgDstMaskPost.jpg", &imgBuffer, TJPF_GRAY, TJSAMP_GRAY, 90);
#endif

	handler->pTransferFunc = (TransformationInfo *)malloc(handler->nPoints * sizeof(TransformationInfo));
	memcpy(handler->pTransferFunc, buffPtr1, handler->nPoints * sizeof(TransformationInfo));
	retval = EXIT_SUCCESS;

cleanup:

#ifndef NDEBUG
	free(imgBuffer.data);
#endif
	free(buffPtr1);
	free(buffPtr2);

	if (retval == EXIT_FAILURE)
		free(handler->pTransferFunc);

	return retval;
}

void processKaleidoscope(KaleidoscopeHandle *handler, double k, ImageData *imgIn, ImageData *imgOut)
{
	// Dim image
	for (unsigned long long idx = 0; idx < imgIn->width * imgIn->height * imgIn->nComponents; ++idx)
		imgOut->data[idx] = (unsigned char)(imgIn->data[idx] * k);
	for (unsigned long long idx = 0; idx < handler->nPoints; ++idx)
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
