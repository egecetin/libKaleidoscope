#include "kaleidoscope.h"

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

int readImage(const char *path, ImageData *img)
{
	// Init variables
	int retval = FAIL;
	int jpegSubsamp = 0, width = 0, height = 0;
	int32_t imgSize = 0;

	uint8_t *compImg = nullptr;
	uint8_t *decompImg = nullptr;

	FILE *fptr = nullptr;
	tjhandle jpegDecompressor = nullptr;

	// Check inputs
	if (!path || !img)
		return FAIL;

	// Find file
	if ((fptr = fopen(path, "rb")) == NULL)
		goto cleanup;
	if (fseek(fptr, 0, SEEK_END) < 0 || ((imgSize = ftell(fptr)) < 0) || fseek(fptr, 0, SEEK_SET) < 0)
		goto cleanup;
	if (imgSize == 0)
		goto cleanup;

	// Read file
	compImg = (uint8_t *)malloc(imgSize * sizeof(uint8_t));
	if (fread(compImg, imgSize, 1, fptr) < 1)
		goto cleanup;

	// Decompress
	jpegDecompressor = tjInitDecompress();
	if (!jpegDecompressor)
		goto cleanup;

	retval = tjDecompressHeader2(jpegDecompressor, compImg, imgSize, &width, &height, &jpegSubsamp);
	if (retval < 0)
		goto cleanup;
	decompImg = (uint8_t *)malloc(width * height * COLOR_COMPONENTS * sizeof(uint8_t));
	retval = tjDecompress2(jpegDecompressor, compImg, imgSize, decompImg, width, 0, height, TJPF_RGB, TJFLAG_FASTDCT);
	if (retval < 0)
		goto cleanup;

	// Set output
	img->width = width;
	img->height = height;
	img->data = decompImg;
	decompImg = nullptr;

cleanup:

	tjDestroy(jpegDecompressor);
	fclose(fptr);

	free(compImg);
	free(decompImg);

	return retval;
}

int saveImage(const char *path, ImageData *img)
{
	// Init variables
	int retval = FAIL;
	uint32_t outSize = 0;

	uint8_t *compImg = nullptr;

	tjhandle jpegCompressor = nullptr;
	FILE *fptr = nullptr;

	// Check inputs
	if (!path || !img)
		return FAIL;

	// Compress
	jpegCompressor = tjInitCompress();
	if (!jpegCompressor)
		goto cleanup;

	retval = tjCompress2(jpegCompressor, img->data, img->width, 0, img->height, TJPF_RGB, &compImg, &outSize, TJSAMP_444, JPEG_QUALITY, TJFLAG_FASTDCT);
	if (retval < 0)
		goto cleanup;

	// Write file
	retval = FAIL; // To simplfy if checks
	if ((fptr = fopen(path, "wb")) == NULL)
		goto cleanup;
	if (fwrite(compImg, outSize, 1, fptr) < 1)
		goto cleanup;

	// Clean ImageData (since the main aim of the app is write data to file)
	retval = SUCCESS;
	free(img->data);
	img->data = nullptr;
	img->height = 0;
	img->width = 0;

cleanup:

	fclose(fptr);
	tjDestroy(jpegCompressor);
	tjFree(compImg);

	return retval;
}

static inline int dimBackground(ImageData *img, double k, ImageData *out)
{
	// Check input
	if (!img)
		return FAIL;

	// Determine whether in-place or out-of-place
	if (!out)
		out = img;
	else
	{
		out->data = (uint8_t *)malloc(img->width * img->height * COLOR_COMPONENTS * sizeof(uint8_t));
		if (!(out->data))
			return FAIL;

		out->width = img->width;
		out->height = img->height;
	}

	uint8_t *ptrIn = img->data;
	uint8_t *ptrOut = out->data;
	const uint64_t len = img->width * img->height * COLOR_COMPONENTS;

	for (uint64_t idx = 0; idx < len; ++idx)
		ptrOut[idx] = (uint8_t)(ptrIn[idx] * k);

	return SUCCESS;
}

static inline int sliceTriangle(ImageData *img, PointData **slicedData, uint64_t *len, int n, double scaleDown)
{
	// Init variables
	uint64_t ctr = 0;
	const float topAngle = 360.0f / n;
	const float quantizationScale = 1.1f;
	const double tanVal = tan(topAngle / 2 * M_PI / 180);

	// Sliced data should be centered before the operation
	const uint32_t preMoveHeight = abs((int32_t)(img->width / (4 * tanVal)) - (int32_t)img->height / 2);
	// Sliced data should be centered after the scaledown
	const uint32_t moveHeight = (uint32_t)(img->height / 2 * scaleDown);

	// Allocate output (Mathematical area differ from pixel area because of quantization)
	*slicedData = (PointData *)malloc((size_t)(img->height * (img->height * tanVal) * COLOR_COMPONENTS * quantizationScale * sizeof(PointData)));
	if (!(*slicedData))
		return FAIL;
	PointData *pSlicedData = *slicedData;

	for (uint32_t idx = 0; idx < img->height - preMoveHeight; ++idx)
	{
		// Fix points
		const uint32_t heightOffset = (idx + preMoveHeight) * img->width * COLOR_COMPONENTS;
		const uint32_t currentHeight = (uint32_t)(round(((int64_t)idx - img->height / 2) * scaleDown) + moveHeight);

		// Offset is the base length / 2 of triangle for current height
		const uint32_t offset = (uint32_t)(idx * tanVal);

		// Calculate indexes
		uint32_t start = (img->width / 2 - offset) * COLOR_COMPONENTS;
		start = start - start % COLOR_COMPONENTS; // Move to first color component
		uint32_t end = start + 2 * offset * COLOR_COMPONENTS;

		for (uint32_t jdx = start; jdx < end; jdx += COLOR_COMPONENTS)
		{
			// Point position respect to center
			pSlicedData[ctr].x = (int32_t)(round(((int64_t)jdx - img->width / 2 * COLOR_COMPONENTS) / COLOR_COMPONENTS * scaleDown));
			pSlicedData[ctr].y = currentHeight;

			// Values of the pixel
			memcpy(pSlicedData[ctr].value, &(img->data[heightOffset + jdx]), COLOR_COMPONENTS);
			ctr += COLOR_COMPONENTS;
		}
	}
	*len = ctr;

	return SUCCESS;
}

static inline int rotateAndMerge(ImageData *img, PointData *slicedData, uint64_t len, uint8_t *hitData, int n)
{
	for (uint16_t idx = 0; idx < n; ++idx)
	{
		double rotationAngle = idx * (360.0 / n);

		// Find rotation matrix
		double cosVal = cos(rotationAngle * M_PI / 180);
		double sinVal = sin(rotationAngle * M_PI / 180);

		// Rotate data and merge
		for (uint64_t jdx = 0; jdx < len; ++jdx)
		{
			// New coordinates (Origin is the center of image)
			int32_t newX = (int32_t)(slicedData[jdx].x * cosVal + slicedData[jdx].y * sinVal);
			int32_t newY = (int32_t)(slicedData[jdx].y * cosVal - slicedData[jdx].x * sinVal);

			// Find absolute coordinates (Origin left top)
			newX = newX + img->width / 2;
			newY = newY + img->height / 2;

			if (newX < img->width && newX > 0 && newY < img->height && newY > 0)
			{
				// Sign point
				hitData[newY * img->width + newX] = 1;
				// Merge
				memcpy(&(img->data[(newY * img->width + newX) * COLOR_COMPONENTS]), slicedData[jdx].value, COLOR_COMPONENTS);
			}
		}
	}

	return SUCCESS;
}

static inline int interpolate(ImageData *img, uint8_t *hitData)
{
	for (uint64_t idx = 1; idx < img->height - 1; ++idx)
	{
		uint64_t heightOffset = idx * img->width;
		for (uint64_t jdx = 1; jdx < img->width - 1; ++jdx)
		{
			if (!hitData[heightOffset + jdx])
			{
				// Check left
				if (hitData[heightOffset + jdx - 1])
					memcpy(&(img->data[(heightOffset + jdx) * COLOR_COMPONENTS]), &(img->data[(heightOffset + jdx - 1) * COLOR_COMPONENTS]), COLOR_COMPONENTS);
				// Check right
				else if (hitData[heightOffset + jdx + 1])
					memcpy(&(img->data[(heightOffset + jdx) * COLOR_COMPONENTS]), &(img->data[(heightOffset + jdx + 1) * COLOR_COMPONENTS]), COLOR_COMPONENTS);
				// Check above
				else if (hitData[heightOffset + jdx - img->width])
					memcpy(&(img->data[(heightOffset + jdx) * COLOR_COMPONENTS]), &(img->data[(heightOffset + jdx - img->width) * COLOR_COMPONENTS]), COLOR_COMPONENTS);
				// Check below
				else if (hitData[heightOffset + jdx + img->width])
					memcpy(&(img->data[(heightOffset + jdx) * COLOR_COMPONENTS]), &(img->data[(heightOffset + jdx + img->width) * COLOR_COMPONENTS]), COLOR_COMPONENTS);
			}
		}
	}

	return SUCCESS;
}

int kaleidoscope(ImageData *img, int n, double k, double scaleDown)
{
	int retval = FAIL;
	uint64_t len = 0;
	PointData *slicedData = nullptr;

	// Check inputs
	if (!img || n < 0 || k < 0.0)
		return FAIL;

	uint8_t *hitData = (uint8_t *)calloc(img->width * img->height, sizeof(uint8_t));
	if (!hitData)
		return FAIL;

	// Slice triangle
	retval = sliceTriangle(img, &slicedData, &len, n, scaleDown);
	if (retval < 0)
		goto cleanup;

	// Prepare background image
	retval = dimBackground(img, k, nullptr);
	if (retval < 0)
		goto cleanup;

	// Rotate and merge with background
	retval = rotateAndMerge(img, slicedData, len, hitData, n);
	if (retval < 0)
		goto cleanup;

	// Interpolate
	retval = interpolate(img, hitData);
	if (retval < 0)
		goto cleanup;

	retval = SUCCESS;
cleanup:

	free(slicedData);
	free(hitData);

	return retval;
}