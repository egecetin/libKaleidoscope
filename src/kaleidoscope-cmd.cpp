#include <chrono>
#include <iostream>
#include <memory>
#include <string>

extern "C"
{
#include <c/kaleidoscope.h>
#include <c/jpeg-utils/jpeg-utils.h>
}

#ifdef KALEIDOSCOPE_ENABLE_CUDA
#include <cuda/kaleidoscope-cuda.hpp>
#endif

int main(int argc, char *argv[])
{
	// Path to images
	std::string path, outPath;

	// Transform handler
	std::shared_ptr<KaleidoscopeHandle> handler;
	// Pointers to image data
	std::unique_ptr<ImageData> inImgData, outImgData;
	// Default number of images to create effect
	int n = 6;
	// Default dim and scaleDown constants
	double k = 0.30, scaleDown = 0.45;
	// Size of the image
	size_t imgSize;

	// Function return value checks
	int retval = EXIT_FAILURE;

	// Benchmark mode variables
	bool benchmark = false;
	size_t maxCtr = 1;
	std::chrono::high_resolution_clock startTime, endTime;

	// CUDA mode variables
	bool useCUDA = false;
	std::unique_ptr<uint8_t> inImgCuda, outImgCuda;
#ifdef KALEIDOSCOPE_ENABLE_CUDA
	cudaStream_t stream;
#endif

	std::cout << "Kaleidoscope Library " << getKaleidoscopeLibraryInfo() << std::endl;

	// Parse and check inputs <----------------------------------------------

	std::cout << "Start ..." << std::endl;

	// Read image and prepare memory
	std::cout << "Reading " << path << " ..." << std::endl;
	if ((retval = readImage(path.c_str(), inImgData.get())))
		return retval;	
	if (initImageData(outImgData.get(), inImgData->width, inImgData->height, inImgData->nComponents))
		return EXIT_FAILURE;
	imgSize = inImgData->width * inImgData->height * inImgData->nComponents * sizeof(uint8_t);

	std::cout << "Initializing transform ..." << std::endl;
	if (useCUDA)
	{
#ifdef KALEIDOSCOPE_ENABLE_CUDA
		// Initialize CUDA stream and prepare transform
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		initKaleidoscopeCuda(handler, k, n, inImgData.width, inImgData.height, inImgData.nComponents, scaleDown, stream);

		initDeviceMemory(inImgCuda.get(), imgSize, stream);
		initDeviceMemory(outImgCuda.get(), imgSize, stream);
		uploadToDeviceImageData(inImgData.data, imgCuda, imgSize, stream);
#else
		useCUDA = false;
		std::cout << "Please enable CUDA backend at compile time" << std::endl;
		return EXIT_FAILURE;
#endif
	}
	else
	{
		if ((retval = initKaleidoscope(&handler, k, n, inImgData.width, inImgData.height, inImgData.nComponents, scaleDown)))
			return retval;
	}

	std::cout << "Processing ..." << std::endl;
	if (benchmark)
		startTime = std::chrono::high_resolution_clock::now();

	// Process image
	for (size_t ctr = 0; ctr < maxCtr; ++ctr)
	{
		if (useCUDA)
#ifdef KALEIDOSCOPE_ENABLE_CUDA
			processKaleidoscopeCuda(handler, inImgCuda.get(), outImgCuda.get(), stream);
#else
			void();
#endif
		else
			processKaleidoscope(handler.get(), inImgData->data, inImgData->data);
	}

	if (benchmark)
		endTime = std::chrono::high_resolution_clock::now();

	std::cout << "Saving " << outPath << " ..." << std::endl;
	if (useCUDA)
		downloadFromDeviceImageData(outImgCuda, outImgData.data, imgSize, stream);
	if ((retval = saveImage(outPath, &outImgData, TJPF_RGB, TJSAMP_444, 90)))
		return retval;

	deInitImageData(&inImgData);
	deInitImageData(&outImgData);



	if (useCUDA)
#ifdef KALEIDOSCOPE_ENABLE_CUDA
		deInitKaleidoscopeCuda(handler, stream);
#else
		void();
#endif
	else
		deInitKaleidoscope(handler.get());

	std::cout << "Done ..." << std::endl;
	return retval;
}
