#ifndef _KALEIDOSCOPE_CUDA_UTILS_CUH_
#define _KALEIDOSCOPE_CUDA_UTILS_CUH_

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

/**
 * @brief Initializes GPU memory for image
 * @param[in, out] ptr Pointer to image array
 * @param[in] siz Required size
 * @param[in] stream CUDA stream
 */
void initDeviceMemory(std::unique_ptr<uint8_t> &ptr, size_t siz, cudaStream_t &stream);

/**
 * @brief Deinitializes GPU memory for image
 * @param[in] ptr Pointer to image array
 * @param[in] stream CUDA stream
 */
void deInitDeviceMemory(std::unique_ptr<uint8_t> &ptr, cudaStream_t &stream);

/**
 * @brief Copy image array to GPU memory
 * @param[in] hostData Host memory
 * @param[out] deviceData GPU memory
 * @param[in] siz Size of image
 * @param[in] stream CUDA stream
 */
void uploadToDeviceImageData(uint8_t *hostData, uint8_t *deviceData, size_t siz, cudaStream_t &stream);

/**
 * @brief Copy image array from GPU memory
 * @param[in] deviceData GPU memory
 * @param[out] hostData Host memory
 * @param[in] siz Size of image
 * @param[in] stream CUDA stream
 */
void downloadFromDeviceImageData(uint8_t *deviceData, uint8_t *hostData, size_t siz, cudaStream_t &stream);

#endif // _KALEIDOSCOPE_CUDA_UTILS_CUH_
