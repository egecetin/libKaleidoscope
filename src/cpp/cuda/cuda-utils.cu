#include "cuda/cuda-utils.hpp"

__host__ void initDeviceMemory(std::unique_ptr<uint8_t> &ptr, size_t siz, cudaStream_t &stream)
{
	cudaMallocAsync((void **)&(ptr.get()), siz, stream);
}

__host__ void deInitDeviceMemory(std::unique_ptr<uint8_t> &ptr, cudaStream_t &stream)
{
	cudaFreeAsync((void *)ptr.get(), stream);
}

__host__ void uploadToDeviceImageData(uint8_t *hostData, uint8_t *deviceData, size_t siz, cudaStream_t &stream)
{
	cudaMemcpyAsync(deviceData, hostData, siz, cudaMemcpyHostToDevice, stream);
}

__host__ void downloadFromDeviceImageData(uint8_t *deviceData, uint8_t *hostData, size_t siz, cudaStream_t &stream)
{
	cudaMemcpyAsync(hostData, deviceData, siz, cudaMemcpyDeviceToHost, stream);
}
