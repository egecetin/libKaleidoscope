#include <kaleidoscope.hpp>

#include <array>
#include <fstream>

#include <gtest/gtest.h>

TEST(CppTests, processingTest)
{
	const double k = 0.30;
	const double scaleDown = 0.45;
	const int n = 6, width = 1935, height = 1088, nComponents = 3;

	std::ifstream inFile("../../tests/data/processing_1935x1088_InputData.bin");
	std::ifstream expectedFile("../../tests/data/processing_1935x1088_ExpectedData.bin");

	const size_t nPixel = width * height * nComponents;
	static std::array<uint8_t, nPixel> inData{0};
	static std::array<uint8_t, nPixel> outData{127};
	static std::array<uint8_t, nPixel> expectedData{255};

	ASSERT_TRUE(inFile.is_open());
	ASSERT_TRUE(expectedFile.is_open());

	inFile.read(reinterpret_cast<char *>(inData.data()), inData.size());

	kalos::Kaleidoscope handle(n, width, height, nComponents, scaleDown, k);
	handle.processImage(inData.data(), outData.data(), nPixel);

	expectedFile.read(reinterpret_cast<char *>(expectedData.data()), expectedData.size());
	ASSERT_TRUE(outData == expectedData);
}
