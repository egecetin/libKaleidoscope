#ifndef _KALEIDOSCOPE_HPP_
#define _KALEIDOSCOPE_HPP_

#include <c/kaleidoscope-definitions.h>

#include <memory>
#include <string>

class KaleidoscopeHandler
{
  private:
	/// Handler data itself
	struct KaleidoscopeHandler_t transformData;

	/// Use CUDA backend?
	bool useCUDA;
	/// Calculated block size for dim operation on GPU
	int blockSizeDim;
	/// Calculated grid size for dim operation on GPU
	int gridSizeDim;
	/// Calculated block size for transform itself on GPU
	int blockSizeTransform;
	/// Calculated grid size for transform itself on GPU
	int gridSizeTransform;

  public:
	/**
	 * @brief Constructs a new Kaleidoscope object
	 * @param[in] k Variable to dim background. Should be between 0.0 and 1.0
	 * @param[in] n Number of images for effect
	 * @param[in] width Image width
	 * @param[in] height Image height
	 * @param[in] nComponents Number of image components (eg 3 for RGB)
	 * @param[in] scaleDown Scale down ratio to shrink image. Must be between 0.0 and 1.0
	 * @param[in] initForCUDA Whether use CUDA backend or not
	 */
	KaleidoscopeHandler(double k, int n, int width, int height, int nComponents, double scaleDown,
						bool initForCUDA = false);
	
	/**
	 * @brief Transform image
	 * @param[in] inImgData Input image
	 * @param[out] outImgData Output image
	 * @param[in] pStream CUDA stream pointer for async processing
	 * @return true If successful
	 * @return false otherwise
	 */
	bool processImage(uint8_t* inImgData, uint8_t* outImgData, void *pStream = nullptr);

	/**
	 * @brief Destroys the Kaleidoscope Handler object
	 */
	~KaleidoscopeHandler();

	/**
	 * @brief Get the Kaleidoscope Library version as integer
	 * @param[out] major Major number
	 * @param[out] minor Minor number
	 * @param[out] patch Patch number
	 */
	static void getKaleidoscopeVersion(int &major, int &minor, int &patch);

	/**
	 * @brief Get the Kaleidoscope Library version as string
	 * @return char* Library version
	 */
	static std::string getKaleidoscopeVersionString();

	/**
	 * @brief Get the Kaleidoscope Library info as string
	 * @return char* Library information
	 */
	static std::string getKaleidoscopeLibraryInfo();
};

#endif // _KALEIDOSCOPE_DEFINITIONS_HPP_
