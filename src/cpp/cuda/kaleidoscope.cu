#include "c/kaleidoscope-definitions.hpp"

__global__ void dimImage(double k, size_t nPoints, uint8_t *imgIn, uint8_t *imgOut)
{
	size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < nPoints)
		*(imgOut + offset) = *(imgIn + offset) * k;
}

__global__ void transformImage(KaleidoscopeHandle *handler, uint8_t *imgIn, uint8_t *imgOut)
{
	size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset >= handler->nPoints)
		return;

	TransformationInfo *infoPtr = handler->pTransferFunc + offset;
	for (int idx = 0; idx < handler->nComponents; ++idx)
		*(imgOut + infoPtr->dstOffset + idx) = *(imgIn + infoPtr->srcOffset + idx);
}

__host__ void initKaleidoscopeCuda(std::shared_ptr<KaleidoscopeHandle> handlerGpu, double k, int n, int width,
								   int height, int nComponents, double scaleDown, cudaStream_t &stream)
{
	if (!initKaleidoscope(handlerGpu.get(), k, n, width, height, nComponents, scaleDown))
	{
		TransformationInfo *ptr;
		cudaMallocAsync((void **)&ptr, sizeof(TransformationInfo) * handlerGpu->nPoints, stream);
		cudaMemcpyAsync(ptr, handlerGpu->pTransferFunc, sizeof(TransformationInfo) * handlerGpu->nPoints,
						cudaMemcpyHostToDevice, stream);

		int blockSize = 0, minGridSize = 0;

		// Calculate kernel size fo dim
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dimImage, 0, width * height * nComponents);
		handlerGpu->blockSizeDim = blockSize;
		handlerGpu->gridSizeDim = (width * height * nComponents + blockSize - 1) / blockSize;

		// Calculate kernel size fo transform itself
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, transformImage, 0, handlerGpu->nPoints);
		handlerGpu->blockSizeTransform = blockSize;
		handlerGpu->gridSizeTransform = (handlerGpu->nPoints + blockSize - 1) / blockSize;

		// Sync and set transfer function
		cudaStreamSynchronize(stream);
		free(handlerGpu->pTransferFunc);
		handlerGpu->pTransferFunc = ptr;
		return;
	}

	fprintf(stderr, "Can't initialize GPU");
	deInitKaleidoscope(handlerGpu.get());
}

__host__ void deInitKaleidoscopeCuda(std::shared_ptr<KaleidoscopeHandle> handlerGpu, cudaStream_t &stream)
{
	cudaFreeAsync(handlerGpu->pTransferFunc, stream);
}

__host__ void processKaleidoscopeCuda(std::shared_ptr<KaleidoscopeHandle> handler, uint8_t *imgIn, uint8_t *imgOut,
									  cudaStream_t &stream)
{
	dimImage<<<handler->gridSizeDim, handler->blockSizeDim, 0, stream>>>(
		handler->k, handler->width * handler->height * handler->nComponents, imgIn, imgOut);
	// transformImage<<<handler->gridSizeTransform, handler->blockSizeTransform, 0, stream>>>(handler, imgIn, imgOut);
	cudaStreamSynchronize(stream);
}
