#include "jpeg-utils/jpeg-utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int readImage(const char *path, ImageData *img)
{
	// Init variables
	int retval = EXIT_FAILURE;

	FILE *fptr = NULL;
	int width = 0, height = 0;
	long imgSize = 0;
	unsigned char nComponent = 3, *compImg = NULL, *decompImg = NULL;
	tjhandle jpegDecompressor = NULL;

	// Check inputs
	assert(path);
	assert(img);

	// Find file and get size
	if ((fptr = fopen(path, "rb")) == NULL)
		goto cleanup;
	if (fseek(fptr, 0, SEEK_END) < 0 || ((imgSize = ftell(fptr)) < 0) || fseek(fptr, 0, SEEK_SET) < 0)
		goto cleanup;
	if (imgSize == 0)
		goto cleanup;

	// Read file
	compImg = (unsigned char *)malloc(imgSize * sizeof(unsigned char));
	if (!compImg || (fread(compImg, imgSize, 1, fptr) < 1))
		goto cleanup;

	// Decompress
	jpegDecompressor = tjInitDecompress();
	if (!jpegDecompressor)
		goto cleanup;

	retval = tjDecompressHeader(jpegDecompressor, compImg, imgSize, &width, &height);
	if (retval < 0)
		goto cleanup;
	decompImg = (unsigned char *)malloc((unsigned long long)width * height * nComponent * sizeof(unsigned char));
	if (!decompImg)
		goto cleanup;
	retval = tjDecompress(jpegDecompressor, compImg, imgSize, decompImg, width, 0, height, nComponent, TJFLAG_FASTDCT);
	if (retval < 0)
		goto cleanup;

	// Set output
	img->width = width;
	img->height = height;
	img->nComponents = nComponent;
	img->data = decompImg;
	decompImg = NULL;

cleanup:
	tjDestroy(jpegDecompressor);
	fclose(fptr);

	free(compImg);
	free(decompImg);

	return retval;
}

int saveImage(const char *path, ImageData *img, enum TJPF pixelFormat, enum TJSAMP samplingFormat, int jpegQuality)
{
	// Init variables
	int retval = EXIT_FAILURE;

	FILE *fptr = NULL;
	long unsigned int outSize = 0;
	unsigned char *compImg = NULL;
	tjhandle jpegCompressor = NULL;

	// Check inputs
	assert(path);
	assert(img);

	// Compress
	jpegCompressor = tjInitCompress();
	if (!jpegCompressor)
		goto cleanup;

	retval = tjCompress2(jpegCompressor, img->data, img->width, 0, img->height, pixelFormat, &compImg, &outSize,
						 samplingFormat, jpegQuality, TJFLAG_FASTDCT);
	if (retval < 0)
		goto cleanup;

	// Write file
	retval = EXIT_FAILURE; // To simplify if checks
	if ((fptr = fopen(path, "wb")) == NULL)
		goto cleanup;
	if (fwrite(compImg, outSize, 1, fptr) < 1)
		goto cleanup;
	if (fflush(fptr))
		goto cleanup;

	retval = EXIT_SUCCESS;

cleanup:
	fclose(fptr);
	tjDestroy(jpegCompressor);
	tjFree(compImg);

	return retval;
}

int initImageData(ImageData *img, int width, int height, int nComponents)
{
	img->data = (unsigned char *)malloc((unsigned long long)width * height * nComponents);
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
	{
		free(img->data);
		img->data = NULL;
	}
}
