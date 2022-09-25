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

static inline int dimBackground(ImageData *img, double k, ImageData *out)
{
	// Check input
	if (!img)
		return EXIT_FAILURE;

	// Determine whether in-place or out-of-place
	if (!out)
		out = img;
	else
	{
		out->data = (unsigned char *)malloc(img->width * img->height * img->nComponent * sizeof(unsigned char));
		if (!(out->data))
			return EXIT_FAILURE;

		out->width = img->width;
		out->height = img->height;
	}

	unsigned char *ptrIn = img->data;
	unsigned char *ptrOut = out->data;
	const unsigned long long len = img->width * img->height * img->nComponent;

	for (unsigned long long idx = 0; idx < len; ++idx)
		ptrOut[idx] = (unsigned char)(ptrIn[idx] * k);

	return EXIT_SUCCESS;
}

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
				if (((ptrIn - 1)->dstLocation.x) || ((ptrIn - 1)->dstLocation.y))
					*ptrOut = *(ptrIn - 1);
				else if (((ptrIn + 1)->dstLocation.x) || ((ptrIn + 1)->dstLocation.y))
					*ptrOut = *(ptrIn + 1);
				else if (((ptrIn - width)->dstLocation.x) || ((ptrIn - width)->dstLocation.y))
					*ptrOut = *(ptrIn - width);
				else if (((ptrIn + width)->dstLocation.x) || ((ptrIn + width)->dstLocation.y))
					*ptrOut = *(ptrIn + width);
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
		if (orgData[idx].dstLocation.x && orgData[idx].dstLocation.y)
		{
			int newX = (int)round(orgData[idx].dstLocation.x * cosVal + orgData[idx].dstLocation.y * sinVal);
			int newY = (int)round(orgData[idx].dstLocation.y * cosVal - orgData[idx].dstLocation.x * sinVal);

			// Fix origin to top left again
			newX += (width / 2);
			newY += (height / 2);

			if (newX < width && newX > 0 && newY < height && newY > 0)
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
	retval = EXIT_SUCCESS;

	// Parameters of triangle
	const double topAngle = 360.0 / n;
	const double tanVal = tan(topAngle / 2.0 * M_PI / 180.0); // tan(topAngle / 2) in radians
	const int triangleHeight = (int)round(width / (2.0 * tanVal));

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
	imgBuffer.nComponent = 1;
	imgBuffer.width = width;
	imgBuffer.height = height;
	imgBuffer.data = (unsigned char *)calloc(nPixels, sizeof(unsigned char));
#endif

	// Ensure limits within image
	if (heightStart < 0 || heightStart > height || heightEnd < 0 || heightEnd > height)
		goto cleanup;

	for (int idx = heightStart; idx <= heightEnd; ++idx)
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
			ptr[jdx].dstLocation.x = (jdx - width / 2) * scaleDown;
			ptr[jdx].dstLocation.y = (idx - heightStart - height / 2) * scaleDown + scaleDownOffset;

			ptr[jdx].length = 1;
		}
	}

#ifndef NDEBUG
	// Save source mask as image
	for (size_t idx = 0; idx < nPixels; ++idx)
	{
		if (buffPtr1[idx].srcLocation.x && buffPtr1[idx].srcLocation.y)
			imgBuffer.data[idx] = 255;
		else
			imgBuffer.data[idx] = 0;
	}
	saveImage("imgSrcMask.jpg", &imgBuffer, 90);
#endif

	// Rotate all points and fix origin to left top
	for (int idx = 0; idx < n; ++idx)
	{
		double rotationAngle = idx * (360.0 / n);
		rotatePoints(buffPtr2, buffPtr1, width, height, rotationAngle);
	}

	// Fill rotation artifacts
	interpolate(buffPtr2, buffPtr1, width, height);

#ifndef NDEBUG
	// Save destination mask as image
	for (size_t idx = 0; idx < nPixels; ++idx)
	{
		if (buffPtr1[idx].dstLocation.x && buffPtr1[idx].dstLocation.y)
			imgBuffer.data[idx] = 255;
		else
			imgBuffer.data[idx] = 0;
	}
	saveImage("imgDstMask.jpg", &imgBuffer, 90);
#endif

	/*
	// Reduction and set to points for handler
	handler->nPoints = 0;
	for (unsigned long long idx = 0; idx < nPixels - 1; ++idx)
	{
		unsigned int ctr = 0;
		TransformationInfo *ptr = &buffPtr2[idx];
		if (!(ptr->srcLocation.x) && !(ptr->srcLocation.y) && !(ptr->dstLocation.x) && !(ptr->dstLocation.y))
			continue;

		for (unsigned long long jdx = idx + 1; jdx < nPixels; ++jdx)
		{
			++ctr;
			if ((ptr->srcLocation.y * width + ptr->srcLocation.x) ==
				(buffPtr2[jdx].srcLocation.y * width + buffPtr2[jdx].srcLocation.x + 1))
				ptr = &buffPtr2[jdx];
			else
				break;
		}

		buffPtr1[handler->nPoints] = buffPtr2[idx];
		buffPtr1[handler->nPoints].length = ctr;
		++(handler->nPoints);
		idx += (ctr - 1);
	}

	handler->pTransferFunc = (TransformationInfo *)malloc(handler->nPoints * sizeof(TransformationInfo));
	memcpy(handler->pTransferFunc, buffPtr1, handler->nPoints * sizeof(TransformationInfo));
	retval = EXIT_SUCCESS;
	*/

cleanup:

#ifndef NDEBUG
	free(imgBuffer.data);
#endif

	handler->nPoints = nPixels;
	handler->pTransferFunc = buffPtr1;

	/*
	free(buffPtr1);
	// free(buffPtr2);

	if (retval == EXIT_FAILURE)
		free(handler->pTransferFunc);
	*/

	return retval;
}

void processKaleidoscope(KaleidoscopeHandle *handler, double k, ImageData *imgIn, ImageData *imgOut)
{
	for (size_t idx = 0; idx < imgIn->width * imgIn->height * imgIn->nComponent; idx += 3)
	{
		if (handler->pTransferFunc[idx / imgIn->nComponent].srcLocation.x &&
			handler->pTransferFunc[idx / imgIn->nComponent].srcLocation.y)
		{
			imgOut->data[idx] = 255;
			imgOut->data[idx + 1] = 255;
			imgOut->data[idx + 2] = 255;
		}
		else
		{
			imgOut->data[idx] = 0;
			imgOut->data[idx + 1] = 0;
			imgOut->data[idx + 2] = 0;
		}
	}
	return;

	for (unsigned long long idx = 0; idx < imgIn->width * imgIn->height * imgIn->nComponent; ++idx)
		imgOut->data[idx] = (unsigned char)(imgIn->data[idx] * k);
	for (unsigned long long idx = 0; idx < handler->nPoints; ++idx)
	{
		unsigned long long srcIdx =
			handler->pTransferFunc[idx].srcLocation.y * imgIn->width + handler->pTransferFunc[idx].srcLocation.x;
		unsigned long long dstIdx =
			handler->pTransferFunc[idx].dstLocation.y * imgIn->width + handler->pTransferFunc[idx].dstLocation.x;
		memcpy(imgOut->data[dstIdx], imgIn->data[srcIdx], handler->pTransferFunc[idx].length);
	}
}

void deInitKaleidoscope(KaleidoscopeHandle *handler) { free(handler->pTransferFunc); }
