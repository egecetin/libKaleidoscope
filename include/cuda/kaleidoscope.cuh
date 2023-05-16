#pragma once

#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

extern "C"
{
#include "kaleidoscope.h"
}

namespace kalos
{
	namespace cuda
	{
		/**
		 * @brief Kaleidoscope effect generator with CUDA backend
		 */
		class Kaleidoscope
		{
		  private:
			double k;
			KaleidoscopeHandle handler;

			std::pair<int, int> dimSizes;
			std::pair<int, int> transformSizes;

		  public:
			/**
			 * @brief Construct a new Kaleidoscope object
			 * @param[in] nImage Number of images for effect
			 * @param[in] width Image width
			 * @param[in] height Image height
			 * @param[in] nComponents Number of color components (eg 3 for RGB)
			 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
			 * @param[in] dimConst Variable to dim background. Should be between 0.0 and 1.0
			 * @param[in] stream CUDA stream
			 */
			Kaleidoscope(int nImage, int width, int height, int nComponents, double scaleDown, double dimConst,
						 cudaStream_t stream = 0);

			/**
			 * @brief Creates kaleidoscope effect
			 * @param[in] inImg GPU allocated input image
			 * @param[out] outImg GPU allocated output image
			 * @param[in] size Size of the images
			 * @param[in] dimConst Variable to dim background. Should be between 0.0 and 1.0
			 * @param[in] stream CUDA stream
			 */
			void processImage(uint8_t *inImg, uint8_t *outImg, size_t nPixels, double dimConst,
							  cudaStream_t stream = 0);

			/**
			 * @brief Creates kaleidoscope effect. Uses dim constant provided in constructor
			 * @param[in] inImg GPU allocated input image
			 * @param[out] outImg GPU allocated output image
			 * @param[in] size Size of the images
			 * @param[in] stream CUDA stream
			 */
			void processImage(uint8_t *inImg, uint8_t *outImg, size_t size, cudaStream_t stream = 0)
			{
				processImage(inImg, outImg, size, k, stream);
			}

			/**
			 * @brief Destroy the Kaleidoscope object. Since there is an internal cudaFree call it is synchronized
			 * with default stream
			 */
			~Kaleidoscope();
		};

	} // namespace cuda
} // namespace kalos
