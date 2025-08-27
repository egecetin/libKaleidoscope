#include "cuda/kaleidoscope.cuh"

#include <stdexcept>

__global__ void dimImage(uint8_t *inImg, uint8_t *outImg, size_t nPoints, double k)
{
	size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < nPoints)
		*(outImg + offset) = *(inImg + offset) * k;
}

__global__ void transformImage(uint8_t *inImg, uint8_t *outImg, int nComponents, TransformationInfo *info,
							   size_t nPoints)
{
	size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset >= nPoints)
		return;

	TransformationInfo *infoPtr = info + offset;
	for (int idx = 0; idx < nComponents; ++idx)
		*(outImg + infoPtr->dstOffset + idx) = *(inImg + infoPtr->srcOffset + idx);
}

__host__ void _processImage(uint8_t *inImg, uint8_t *outImg, size_t nPixels, double dimConst, int nComponents,
							TransformationInfo *pFunc, size_t nFunc, std::pair<int, int> dimSizes,
							std::pair<int, int> transformSizes, cudaStream_t stream)
{
	dimImage<<<dimSizes.second, dimSizes.first, 0, stream>>>(inImg, outImg, nPixels, dimConst);
	transformImage<<<transformSizes.second, transformSizes.first, 0, stream>>>(inImg, outImg, nComponents, pFunc,
																			   nFunc);
}

namespace kalos
{
	namespace cuda
	{
		Kaleidoscope::Kaleidoscope(int nImage, int width, int height, int nComponents, double scaleDown,
								   double dimConst)
			: k(dimConst)
		{
			KaleidoscopeHandle handlerLocal;
			if (initKaleidoscope(&handlerLocal, nImage, width, height, nComponents, scaleDown) != 0)
				throw std::runtime_error("Unknown error");

			// Calculate kernel sizes
			int blockSize = 0, minGridSize = 0;
			size_t nPixel = width * height * nComponents;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dimImage, 0, nPixel);
			if (blockSize)
				dimSizes = std::pair<int, int>(blockSize, (nPixel + blockSize - 1) / blockSize);

			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, transformImage, 0, handlerLocal.nPoints);
			if (blockSize)
				transformSizes = std::pair<int, int>(blockSize, (handlerLocal.nPoints + blockSize - 1) / blockSize);

			// Move transform information
			handler.width = handlerLocal.width;
			handler.height = handlerLocal.height;
			handler.nComponents = handlerLocal.nComponents;
			handler.nPoints = handlerLocal.nPoints;

			cudaMalloc(&(handler.pTransferFunc), sizeof(TransformationInfo) * handlerLocal.nPoints);
			cudaMemcpy(handler.pTransferFunc, handlerLocal.pTransferFunc,
					   sizeof(TransformationInfo) * handlerLocal.nPoints, cudaMemcpyHostToDevice);

			deInitKaleidoscope(&handlerLocal);
		}

		void Kaleidoscope::processImage(uint8_t *inImg, uint8_t *outImg, size_t nPixels, double dimConst,
										cudaStream_t stream)
		{
			_processImage(inImg, outImg, nPixels, dimConst, handler.nComponents, handler.pTransferFunc, handler.nPoints,
						  dimSizes, transformSizes, stream);
		}

		Kaleidoscope::~Kaleidoscope() { cudaFree(handler.pTransferFunc); }

	} // namespace cuda
} // namespace kalos
