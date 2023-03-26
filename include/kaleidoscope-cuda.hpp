#ifndef _KALEIDOSCOPE_CUDA_H_
#define _KALEIDOSCOPE_CUDA_H_

#include "kaleidoscope-definitions.h"

#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Initializes kaleidoscope handler
 * @param[in, out] handler Kaleidoscope effect handler
 * @param[in] k Variable to dim background. Should be between 0.0 and 1.0
 * @param[in] n Number of images for effect
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] nComponents Number of image components (eg 3 for RGB)
 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
 * @param[in] stream CUDA stream
 */
void initKaleidoscopeCuda(KaleidoscopeHandle *handlerGpu, double k, int n, int width, int height,
								   int nComponents, double scaleDown, cudaStream_t &stream);

/**
 * @brief Deinitializes kaleidoscope handler
 * @param[in] handler Kaleidoscope effect handler
 * @param[in] stream CUDA stream
 */
void deInitKaleidoscopeCuda(KaleidoscopeHandle *handlerGpu, cudaStream_t &stream);

/**
 * @brief Applies kaleidoscope effect to image
 * @param[in] handler Kaleidoscope effect handler
 * @param[in] imgIn Input image data from device memory (Allocated with cudaMalloc etc)
 * @param[out] imgOut Output image data from device memory (Allocated with cudaMalloc etc)
 * @param[in] stream CUDA stream
 */
void processKaleidoscopeCuda(KaleidoscopeHandle *handler, unsigned char *imgIn, unsigned char *imgOut,
									 cudaStream_t &stream);

/**
 * @brief Initializes GPU memory for image
 * @param[in, out] ptr Pointer to image array
 * @param[in] siz Required size
 * @param[in] stream CUDA stream
 */
void initDeviceMemory(unsigned char **ptr, unsigned long long siz, cudaStream_t &stream);

/**
 * @brief Copy image array to GPU memory
 * @param[in] hostData Host memory
 * @param[out] deviceData GPU memory
 * @param[in] siz Size of image
 * @param[in] stream CUDA stream
 */
void uploadToDeviceImageData(unsigned char *hostData, unsigned char *deviceData, unsigned long long siz, cudaStream_t &stream);

/**
 * @brief Copy image array from GPU memory
 * @param[in] deviceData GPU memory
 * @param[out] hostData Host memory
 * @param[in] siz Size of image
 * @param[in] stream CUDA stream
 */
void downloadFromDeviceImageData(unsigned char *deviceData, unsigned char *hostData, unsigned long long siz, cudaStream_t &stream);

#endif // _KALEIDOSCOPE_CUDA_H_
