#include <kaleidoscope.hpp>

#include <array>
#include <fstream>

#include <gtest/gtest.h>

TEST(CppTests, processingTest)
{
	const double k = 0.30;
	const double scaleDown = 0.45;
	const int n = 6, width = 1935, height = 1088, nComponents = 3;

	std::ifstream inFile("../../tests/data/processing_1935x1088_InputData.bin", std::ios_base::binary);
	std::ifstream expectedFile("../../tests/data/processing_1935x1088_ExpectedData.bin", std::ios_base::binary);

	const size_t nPixel = width * height * nComponents;
	static std::array<uint8_t, nPixel> inData{0};
	static std::array<uint8_t, nPixel> outData{127};
	static std::array<uint8_t, nPixel> expectedData{255};

	ASSERT_TRUE(inFile.is_open());
	ASSERT_TRUE(expectedFile.is_open());

	ASSERT_TRUE(inFile.read(reinterpret_cast<char *>(inData.data()), inData.size()).good());

	std::unique_ptr<kalos::Kaleidoscope> handle;

	ASSERT_NO_THROW(handle = std::make_unique<kalos::Kaleidoscope>(n, width, height, nComponents, scaleDown, k));
	handle->processImage(inData.data(), outData.data(), nPixel);

	ASSERT_TRUE(expectedFile.read(reinterpret_cast<char *>(expectedData.data()), expectedData.size()).good());
	ASSERT_TRUE(outData == expectedData);
}

TEST(CppTests, generalTest)
{
	const double k = 0.30;
	const double scaleDown = 0.45;
	const int n = 6, width = 1935, height = 1088, nComponents = 3;

	ASSERT_NO_THROW(kalos::Kaleidoscope(n, width, height, nComponents, scaleDown, k));

	ASSERT_THROW(kalos::Kaleidoscope(2, width, height, nComponents, scaleDown, k), std::runtime_error);

	ASSERT_THROW(kalos::Kaleidoscope(n, 0, height, nComponents, scaleDown, k), std::runtime_error);
	ASSERT_THROW(kalos::Kaleidoscope(n, -1, height, nComponents, scaleDown, k), std::runtime_error);

	ASSERT_THROW(kalos::Kaleidoscope(n, width, 0, nComponents, scaleDown, k), std::runtime_error);
	ASSERT_THROW(kalos::Kaleidoscope(n, width, -1, nComponents, scaleDown, k), std::runtime_error);

	ASSERT_THROW(kalos::Kaleidoscope(n, width, height, 0, scaleDown, k), std::runtime_error);
	ASSERT_THROW(kalos::Kaleidoscope(n, width, height, -1, scaleDown, k), std::runtime_error);

	ASSERT_THROW(kalos::Kaleidoscope(n, width, height, nComponents, -0.01, k), std::runtime_error);
	ASSERT_THROW(kalos::Kaleidoscope(n, width, height, nComponents, 1.01, k), std::runtime_error);
}
