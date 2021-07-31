#include "header.h"

int readImage(const char *path, ImageData *img)
{
    // Check inputs
    if (!path || !img)
        return FAIL;

    // Init variables
    int retval = FAIL;
    int jpegSubsamp = 0, width = 0, height = 0;
    uint64_t imgSize = 0;

    uint8_t *compImg = nullptr;
    uint8_t *decompImg = nullptr;

    FILE *fptr = nullptr;
    tjhandle jpegDecompressor = nullptr;

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
    img->size = imgSize;
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
    // Check inputs
    if (!path || !img)
        return FAIL;

    // Init variables
    int retval = FAIL;
    uint64_t outSize = 0;

    uint8_t *compImg = nullptr;

    tjhandle jpegCompressor = nullptr;
    FILE *fptr = nullptr;

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
    img->size = 0;

cleanup:

    fclose(fptr);
    tjDestroy(jpegCompressor);
    tjFree(compImg);

    return retval;
}

int dimBackground(ImageData *img, float k, ImageData *out)
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
        out->size = img->size;
    }

    uint8_t *ptrIn = img->data;
    uint8_t *ptrOut = out->data;
    uint64_t len = img->width * img->height * COLOR_COMPONENTS;

#pragma omp simd
    for (uint64_t idx = 0; idx < len; ++idx)
        ptrOut[idx] = ptrIn[idx] * k;

    return 0;
}

int sliceTriangle(ImageData *img, PointData *slicedData, int n, float scaleDown)
{
    // Init variables
    uint64_t ctr = 0;
    uint16_t topAngle = 360 / n;
    double tanVal = tan(topAngle / 2 * M_PI / 180);

    // Sliced data should be centered after the operation
    const uint32_t moveHeight = img->height / 2 * scaleDown;

    // Allocate output
    slicedData = (PointData *)malloc(img->height * (img->height * tanVal) * COLOR_COMPONENTS * sizeof(PointData));
    if (!slicedData)
        return FAIL;

    for (uint32_t idx = 0; idx < img->height; ++idx)
    {
        // Fix points
        uint32_t heightOffset = idx * img->width * COLOR_COMPONENTS;
        uint32_t currentHeight = round((idx - img->height / 2) * scaleDown) + moveHeight;

        // Offset is the base length / 2 of triangle for current height
        uint32_t offset = idx * tanVal;

        // Calculate indexes
        uint32_t start = (img->width / 2 - offset) * COLOR_COMPONENTS;
        start = start - start % 3; // Move to first color component
        uint32_t end = start + 2 * offset * COLOR_COMPONENTS;

        for (uint32_t jdx = start; jdx < end; jdx += 3)
        {
            // Point position respect to center
            slicedData[ctr].x = round((jdx - img->width / 2) * scaleDown);
            slicedData[ctr].y = currentHeight;

            // Values of the pixel
            memcpy(slicedData[ctr].value, img->data[heightOffset + jdx], 3);
            ctr += 3;
        }
    }

    // DEBUG
    uint64_t ctr2 = (img->height * (img->height * tanVal) * COLOR_COMPONENTS);
    printf("Allocated: %ld Set: %ld\n", ctr2, ctr);

    return SUCCESS;
}

int kaleidoscope(ImageData *img, int n, float k, float scaleDown)
{
    if (!img || n < 2 || scaleDown > 0.5)
        return FAIL;

    int retval = FAIL;
    ImageData background;
    PointData *slicedData = nullptr;

    // Prepare background image
    retval = dimBackground(img, 0.5, &background);
    if (retval < 0)
        goto cleanup;

    // Slice triangle
    retval = sliceTriangle(img, slicedData, n, scaleDown);
    if (retval < 0)
        goto cleanup;

    // Rotate and merge with background
    for (int idx = 0; idx < n; ++idx)
    {
        // Find rotation matrix

        // Rotate data and merge
    }

    retval = SUCCESS;
cleanup:

    free(slicedData);
    free(background.data);

    return retval;
}