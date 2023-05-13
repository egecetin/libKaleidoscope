#include "kaleidoscope.cuh"

namespace kalos
{
	namespace cuda
	{
		__device__ void dimImage(uint8_t *inImg, uint8_t *outImg, size_t nPoints, double k)
		{
			size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
			if (offset < nPoints)
				*(outImg + offset) = *(inImg + offset) * k;
		}

		__device__ void transformImage(uint8_t *inImg, uint8_t *outImg, int nComponents, TransformationInfo *info, size_t nPoints)
		{
			size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
			if (offset >= nPoints)
				return;

			TransformationInfo *infoPtr = info + offset;
			for (int idx = 0; idx < nComponents; ++idx)
				*(outImg + infoPtr->dstOffset + idx) = *(inImg + infoPtr->srcOffset + idx);
		}

		void Kaleidoscope::init(int nImage, int width, int height, int nComponents, double scaleDown, double dimConst,
								cudaStream_t stream)
			: k(dimConst)
		{
			KaleidoscopeHandle handlerLocal;
			if (initKaleidoscope(&handler, nImage, width, height, nComponents, scaleDown) != 0)
				throw std::runtime_error("Unknown error");

			// Calculate kernel sizes
			int blockSize = 0, minGridSize = 0;
			size_t nPixel = width * height * nComponents;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dimImage, 0, nPixel);
			dimSizes = std::pair<int, int>(blockSize, (nPixel + blockSize - 1) / blockSize);

			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, transformImage, 0, handler.nPoints);
			transformSizes = std::pair<int, int>(blockSize, (handler.nPoints + blockSize - 1) / blockSize);

			// Move transform information
			handler.width = handlerLocal.width;
			handler.height = handlerLocal.height;
			handler.nComponents = handlerLocal.nComponents;
			handler.nPoints = handlerLocal.nPoints;

			cudaMallocAsync(&(handler.pTransferFunc), sizeof(TransformationInfo) * handlerLocal.nPoints, stream);
			cudaMemcpyAsync(handler.pTransferFunc, handlerLocal.pTransferFunc,
							sizeof(TransformationInfo) * handlerLocal.nPoints, cudaMemcpyHostToDevice, stream);

			deInitKaleidoscope(&handlerLocal);
		}

		__global__ void processImage(uint8_t *inImg, uint8_t *outImg, size_t nPixels, double dimConst,
									 cudaStream_t stream)
		{
			dimImage<<<dimSizes.second, dimSize.first, 0, stream>>>(inImg, outImg, nPixels, dimConst);
			// transformImage<<<transformSizes.second, transformSizes.first, 0, stream>>>(inImg, outImg, handler.nComponents,
			// handler.pTransferFunc, handler.nPoints);
		}

		Kaleidoscope::~Kaleidoscope() { cudaFree(handler.pTransferFunc); }

	} // namespace cuda
} // namespace kalos
