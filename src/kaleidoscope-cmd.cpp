#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

extern "C"
{
#include <c/jpeg-utils/jpeg-utils.h>
}

#include <cpp/kaleidoscope.hpp>

void printUsage()
{
	std::cout << "Usage: ./kaleidoscope <Parameters>" << std::endl;
	std::cout << "" << std::endl;
	std::cout << "Parameters" << std::endl;
	std::cout << "\t"
			  << "--cuda: Use CUDA backend if available" << std::endl;
	std::cout << "\t"
			  << "--help: Displays this info" << std::endl;
	std::cout << "\t"
			  << "--in  : Input image path" << std::endl;
	std::cout << "\t"
			  << "--out : Output image path" << std::endl;
	std::cout << "\t"
			  << "-b    : Benchmark mode. Provide a counter value to measure performance" << std::endl;
	std::cout << "\t"
			  << "-k    : Dim constant for background" << std::endl;
	std::cout << "\t"
			  << "-N    : Number of images for effect" << std::endl;
	std::cout << "\t"
			  << "-s    : Scale down constant" << std::endl;
}

/**
 * @brief Parses command line inputs
 */
class InputParser
{
  public:
	/**
	 * @brief Constructs a new InputParser object
	 * @param[in] argc Number of input arguments
	 * @param[in] argv Input arguments
	 */
	InputParser(const int &argc, char **argv)
	{
		for (int i = 1; i < argc; ++i)
			this->tokens.push_back(std::string(argv[i]));
		this->tokens.push_back(std::string(""));
	}

	/**
	 * @brief Gets single command line input
	 * @param[in] option Option to check
	 * @return const std::string& Found command line input. Empty string if not found
	 */
	const std::string &getCmdOption(const std::string &option) const
	{
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->tokens.begin(), this->tokens.end(), option);
		if (itr != this->tokens.end() && ++itr != this->tokens.end())
		{
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}

	/**
	 * @brief Checks whether provided command line option is exists.
	 * @param[in] option Option to check
	 * @return true If the provided option is found
	 * @return false If the provided option is not found
	 */
	bool cmdOptionExists(const std::string &option) const
	{
		return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
	}

  private:
	std::vector<std::string> tokens;
};

int main(int argc, char *argv[])
{
	// Path to images
	std::string inPath, outPath;

	// Transform handler
	std::shared_ptr<KaleidoscopeHandler> handler;
	// Pointers to image data
	std::unique_ptr<ImageData> inImgData, outImgData;
	// Default number of images to create effect
	int n = 6;
	// Default dim and scaleDown constants
	double k = 0.30, scaleDown = 0.45;
	// Size of the image
	size_t imgSize;
	// Image dimensions
	int width, height, nComponents;

	// Function return value checks
	int retval = EXIT_FAILURE;

	// Benchmark mode variables
	bool benchmark = false;
	size_t maxCtr = 1;
	std::chrono::high_resolution_clock::time_point startTime, endTime;

	// CUDA mode variables
	bool useCUDA = false;

	std::cout << "Kaleidoscope Library " << KaleidoscopeHandler::getKaleidoscopeLibraryInfo() << std::endl;

	// Parse and check inputs
	{
		InputParser input(argc, argv);

		if (input.cmdOptionExists("--help"))
		{
			printUsage();
			return EXIT_SUCCESS;
		}

		if (!input.cmdOptionExists("--in") || !input.cmdOptionExists("--out"))
		{
			std::cout << "You should provide an input and output file" << std::endl;
			printUsage();
			return EXIT_FAILURE;
		}

		inPath = input.getCmdOption("--in");
		outPath = input.getCmdOption("--out");

		if (input.cmdOptionExists("-N"))
		{
			try
			{
				n = std::stoi(input.getCmdOption("-N"));
				if (n <= 0)
					throw std::runtime_error("N is negative");
			}
			catch (const std::exception &e)
			{
				std::cout << "Provide N as a positive valid integer" << std::endl;
				return EXIT_FAILURE;
			}
		}

		if (input.cmdOptionExists("-k"))
		{
			try
			{
				k = std::stof(input.getCmdOption("-k"));
				if (k < 0)
					throw std::runtime_error("k is negative");
			}
			catch (const std::exception &e)
			{
				std::cout << "Provide k as a positive valid value" << std::endl;
				return EXIT_FAILURE;
			}
		}

		if (input.cmdOptionExists("-s"))
		{
			try
			{
				scaleDown = std::stof(input.getCmdOption("-s"));
				if (scaleDown < 0)
					throw std::runtime_error("Scale down parameter is negative");
			}
			catch (const std::exception &e)
			{
				std::cout << "Provide scaleDown parameter as a positive valid value" << std::endl;
				return EXIT_FAILURE;
			}
		}

		if (input.cmdOptionExists("-b"))
		{
			try
			{
				maxCtr = std::stoull(input.getCmdOption("-b"));
				benchmark = true;
			}
			catch (const std::exception &e)
			{
				std::cout << "Provide valid parameter for benchmark counter" << std::endl;
				return EXIT_FAILURE;
			}
		}

		if (input.cmdOptionExists("--cuda"))
			useCUDA = true;
	}

	std::cout << "Start ..." << std::endl;

	// Read image and prepare memory
	std::cout << "Reading " << inPath << " ..." << std::endl;
	if ((retval = readImage(inPath.c_str(), inImgData.get())))
		return retval;

	width = inImgData->width;
	height = inImgData->height;
	nComponents = inImgData->nComponents;
	imgSize = width * height * nComponents * sizeof(uint8_t);

	if (initImageData(outImgData.get(), width, height, nComponents))
		return EXIT_FAILURE;

	std::cout << "Initializing transform ..." << std::endl;
	handler = std::make_shared<KaleidoscopeHandler>(k, n, width, height, nComponents, scaleDown, useCUDA);

	std::cout << "Processing ..." << std::endl;
	if (benchmark)
		startTime = std::chrono::high_resolution_clock::now();

	// Process image
	for (size_t ctr = 0; ctr < maxCtr; ++ctr)
		handler->processImage(inImgData->data, outImgData->data);

	if (benchmark)
		endTime = std::chrono::high_resolution_clock::now();

	std::cout << "Saving " << outPath << " ..." << std::endl;
	if ((retval = saveImage(outPath.c_str(), outImgData.get(), TJPF_RGB, TJSAMP_444, 90)))
		return retval;

	deInitImageData(inImgData.get());
	deInitImageData(outImgData.get());

	std::cout << "Done ..." << std::endl;
	if (benchmark)
		std::cout << static_cast<double>(maxCtr) / (endTime - startTime).count() << " FPS" << std::endl;
	return retval;
}
