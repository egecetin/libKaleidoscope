#include <cuda/kaleidoscope.cuh>

#include <array>
#include <fstream>

#include <gtest/gtest.h>

TEST(CppTests, processingTestCuda)
{
	const double k = 0.30;
	const double scaleDown = 0.45;
	const int n = 6, width = 1935, height = 1088, nComponents = 3;

	std::ifstream inFile("../../tests/data/processing_1935x1088_InputData.bin");
	std::ifstream expectedFile("../../tests/data/processing_1935x1088_ExpectedData.bin");

    const size_t nPixel = width * height * nComponents;
	static std::array<uint8_t, nPixel> inData;
	static std::array<uint8_t, nPixel> outData;
	static std::array<uint8_t, nPixel> expectedData;

	inFile.read(reinterpret_cast<char *>(inData.data()), inData.size());

    uint8_t *deviceInData = nullptr;
    uint8_t *deviceOutData = nullptr;

    // Upload data to device
    cudaMalloc(&deviceInData, nPixel);
    cudaMalloc(&deviceOutData, nPixel);
    cudaMemcpy(deviceInData, inData.data(), nPixel, cudaMemcpyHostToDevice);

	kalos::cuda::Kaleidoscope handle(n, width, height, nComponents, scaleDown, k);
	handle.processImage(deviceInData, deviceOutData, nPixel);

	// Download data from device
    cudaMemcpy(outData.data(), deviceOutData, nPixel, cudaMemcpyDeviceToHost);
    cudaFree(deviceInData);
    cudaFree(deviceOutData);

	std::ofstream outFile("cudaOutput.dat", std::ios_base::binary);
	outFile.write(reinterpret_cast<char *>(outData.data()), outData.size());
	outFile.close();

    expectedFile.read(reinterpret_cast<char *>(expectedData.data()), expectedData.size());
	ASSERT_TRUE(outData == expectedData);
}
