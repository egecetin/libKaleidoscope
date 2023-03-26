#include "kaleidoscope-cuda.hpp"

#include <cstdio>

extern "C"
{
#include "kaleidoscope.h"
}

__global__ void dimImage(double k, size_t nPoints, unsigned char *imgIn, unsigned char *imgOut)
{
	size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < nPoints)
		*(imgOut + offset) = *(imgIn + offset) * k;
}

__global__ void transformImage(KaleidoscopeHandle *handler, unsigned char *imgIn, unsigned char *imgOut)
{
	size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
	printf("%d : %x\n", offset, handler);
	if (offset >= handler->nPoints)
		return;

	TransformationInfo *infoPtr = handler->pTransferFunc + offset;
	printf("%d %d\n", infoPtr->dstOffset, infoPtr->srcOffset);
	for (int idx = 0; idx < handler->nComponents; ++idx)
		*(imgOut + infoPtr->dstOffset + idx) = *(imgIn + infoPtr->srcOffset + idx);
}

__host__ void initKaleidoscopeCuda(KaleidoscopeHandle *handlerGpu, double k, int n, int width, int height,
								   int nComponents, double scaleDown, cudaStream_t &stream)
{
	if (!initKaleidoscope(handlerGpu, k, n, width, height, nComponents, scaleDown))
	{
		TransformationInfo *ptr;
		cudaMallocAsync((void **)&ptr, sizeof(TransformationInfo) * handlerGpu->nPoints, stream);
		cudaMemcpyAsync(ptr, handlerGpu->pTransferFunc, sizeof(TransformationInfo) * handlerGpu->nPoints,
						cudaMemcpyHostToDevice, stream);
		cudaStreamSynchronize(stream);

		free(handlerGpu->pTransferFunc);
		handlerGpu->pTransferFunc = ptr;

		// Calculate kernel sizes
		int minGridSize = 0;
		int blockSize = 0;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dimImage, 0, width * height * nComponents);
		handlerGpu->blockSizeDim = blockSize;
		handlerGpu->gridSizeDim = (width * height * nComponents + blockSize - 1) / blockSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, transformImage, 0, handlerGpu->nPoints);
		handlerGpu->blockSizeTransform = blockSize;
		handlerGpu->gridSizeTransform = (handlerGpu->nPoints + blockSize - 1) / blockSize;

		return;
	}

	fprintf(stderr, "Can't initialize GPU");
	deInitKaleidoscope(handlerGpu);
}

__host__ void deInitKaleidoscopeCuda(KaleidoscopeHandle *handlerGpu, cudaStream_t &stream)
{
	cudaFreeAsync(handlerGpu->pTransferFunc, stream);
	cudaStreamSynchronize(stream);
	handlerGpu->pTransferFunc = nullptr;
}

__host__ void processKaleidoscopeCuda(KaleidoscopeHandle *handler, unsigned char *imgIn, unsigned char *imgOut,
									  cudaStream_t &stream)
{
	dimImage<<<handler->gridSizeDim, handler->blockSizeDim, 0, stream>>>(
		handler->k, handler->width * handler->height * handler->nComponents, imgIn, imgOut);
	transformImage<<<handler->gridSizeTransform, handler->blockSizeTransform, 0, stream>>>(handler, imgIn, imgOut);
	cudaStreamSynchronize(stream);
}

__host__ void initDeviceMemory(unsigned char **ptr, unsigned long long siz, cudaStream_t &stream)
{
	cudaMallocAsync((void **)ptr, siz, stream);
	cudaStreamSynchronize(stream);
}

__host__ void uploadToDeviceImageData(unsigned char *hostData, unsigned char *deviceData, unsigned long long siz,
									  cudaStream_t &stream)
{
	cudaMemcpyAsync(deviceData, hostData, siz, cudaMemcpyHostToDevice, stream);
	cudaStreamSynchronize(stream);
}

__host__ void downloadFromDeviceImageData(unsigned char *deviceData, unsigned char *hostData, unsigned long long siz,
										  cudaStream_t &stream)
{
	cudaMemcpyAsync(hostData, deviceData, siz, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
}
