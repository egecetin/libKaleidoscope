#include "JpegUtils.h"

#include <stdio.h>
#include <stdlib.h>

#include <turbojpeg.h>

int readImage(const char *path, ImageData *img)
{
	// Init variables
	int retval = EXIT_FAILURE;

	int width = 0, height = 0;
	unsigned long imgSize = 0;
	unsigned char nComponent = 3;

	unsigned char *compImg = NULL;
	unsigned char *decompImg = NULL;

	FILE *fptr = NULL;
	tjhandle jpegDecompressor = NULL;

	// Check inputs
	if (!path || !img)
		return EXIT_FAILURE;

	// Find file and get size
	if ((fptr = fopen(path, "rb")) == NULL)
		goto cleanup;
	if (fseek(fptr, 0, SEEK_END) < 0 || ((imgSize = ftell(fptr)) < 0) || fseek(fptr, 0, SEEK_SET) < 0)
		goto cleanup;
	if (imgSize == 0)
		goto cleanup;

	// Read file
	compImg = (unsigned char *)malloc(imgSize * sizeof(unsigned char));
	if (fread(compImg, imgSize, 1, fptr) < 1)
		goto cleanup;

	// Decompress
	jpegDecompressor = tjInitDecompress();
	if (!jpegDecompressor)
		goto cleanup;

	retval = tjDecompressHeader(jpegDecompressor, compImg, imgSize, &width, &height);
	if (retval < 0)
		goto cleanup;
	decompImg = (unsigned char *)malloc(width * height * nComponent * sizeof(unsigned char));
	retval = tjDecompress(jpegDecompressor, compImg, imgSize, decompImg, width, 0, height, TJPF_RGB, TJFLAG_FASTDCT);
	if (retval < 0)
		goto cleanup;

	// Set output
	img->width = width;
	img->height = height;
	img->nComponent = nComponent;
	img->format = TJPF_RGB;
	img->data = decompImg;
	decompImg = NULL;

cleanup:
	tjDestroy(jpegDecompressor);
	fclose(fptr);

	free(compImg);
	free(decompImg);

	return retval;
}

int saveImage(const char *path, ImageData *img, int jpegQuality)
{
	// Init variables
	int retval = EXIT_FAILURE;
	uint32_t outSize = 0;

	FILE *fptr = NULL;
	uint8_t *compImg = NULL;
	tjhandle jpegCompressor = NULL;

	// Check inputs
	if (!path || !img)
		return EXIT_FAILURE;

	// Compress
	jpegCompressor = tjInitCompress();
	if (!jpegCompressor)
		goto cleanup;

	retval = tjCompress(jpegCompressor, img->data, img->width, 0, img->height, img->format, &compImg, &outSize,
						TJSAMP_444, jpegQuality, TJFLAG_FASTDCT);
	if (retval < 0)
		goto cleanup;

	// Write file
	retval = EXIT_FAILURE; // To simplfy if checks
	if ((fptr = fopen(path, "wb")) == NULL)
		goto cleanup;
	if (fwrite(compImg, outSize, 1, fptr) < 1)
		goto cleanup;

	// Clean ImageData (since the main aim of the app is write data to file)
	retval = EXIT_SUCCESS;

cleanup:
	fclose(fptr);
	tjDestroy(jpegCompressor);
	tjFree(compImg);

	return retval;
}