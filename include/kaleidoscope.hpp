#pragma once

extern "C"
{
#include "kaleidoscope.h"
}

#include <stdexcept>

namespace kalos
{
	/**
	 * @brief Kaleidoscope effect generator
	 */
	class Kaleidoscope
	{
	  private:
		double k;
		KaleidoscopeHandle handler;

	  public:
		/**
		 * @brief Construct a new Kaleidoscope object
		 * @param[in] nImage Number of images for effect
		 * @param[in] width Image width
		 * @param[in] height Image height
		 * @param[in] nComponents Number of color components (eg 3 for RGB)
		 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
		 * @param[in] dimConst Variable to dim background. Should be between 0.0 and 1.0
		 */
		Kaleidoscope(int nImage, int width, int height, int nComponents, double scaleDown, double dimConst)
			: k(dimConst)
		{
			if (initKaleidoscope(&handler, nImage, width, height, nComponents, scaleDown) != 0)
				throw std::runtime_error("Can't init kaleidoscope structure for these inputs");
		}

		/**
		 * @brief Creates kaleidoscope effect
		 * @param inImg Input image
		 * @param outImg Output image
		 * @param size Size of the images
		 * @param dimConst Variable to dim background. Should be between 0.0 and 1.0
		 */
		void processImage(uint8_t *inImg, uint8_t *outImg, size_t size, double dimConst)
		{
			processKaleidoscope(&handler, dimConst, inImg, outImg);
		}

		/**
		 * @brief Creates kaleidoscope effect. Uses dim constant provided in constructor
		 * @param inImg Input image
		 * @param outImg Output image
		 * @param size Size of the images
		 */
		void processImage(uint8_t *inImg, uint8_t *outImg, size_t size) { processImage(inImg, outImg, size, k); }

		/**
		 * @brief Destroy the Kaleidoscope object
		 */
		~Kaleidoscope() { deInitKaleidoscope(&handler); }
	};

} // namespace kalos
