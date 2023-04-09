#include <cpp/kaleidoscope.hpp>
#include <kaleidoscope-config.h>

#include <stdexcept>

extern "C"
{
#include <c/kaleidoscope.h>
}

KaleidoscopeHandler::KaleidoscopeHandler(double k, int n, int width, int height, int nComponents, double scaleDown,
										 bool initForCUDA)
{
	if (initKaleidoscope(&transformData, k, n, width, height, nComponents, scaleDown))
		throw std::runtime_error("Cant init kaleidoscope handler");
}

bool KaleidoscopeHandler::processImage(uint8_t *inImgData, uint8_t *outImgData, void *pStream)
{
	processKaleidoscope(&transformData, inImgData, outImgData);
	return true;
}

KaleidoscopeHandler::~KaleidoscopeHandler() { deInitKaleidoscope(&transformData); }

void KaleidoscopeHandler::getKaleidoscopeVersion(int &major, int &minor, int &patch)
{
	major = PROJECT_MAJOR_VERSION;
	minor = PROJECT_MINOR_VERSION;
	patch = PROJECT_PATCH_VERSION;
}

std::string KaleidoscopeHandler::getKaleidoscopeVersionString() { return PROJECT_VERSION; }

std::string KaleidoscopeHandler::getKaleidoscopeLibraryInfo()
{
	std::string info = std::string(PROJECT_VERSION) + " " + C_COMPILER_NAME + " " + C_COMPILER_VERSION + " " +
					   CXX_COMPILER_NAME + " " + CXX_COMPILER_VERSION + " " + BUILD_TYPE + " " + PROJECT_BUILD_DATE +
					   " " + PROJECT_BUILD_TIME + " ";
#ifdef KALEIDOSCOPE_ENABLE_CUDA
	info = info + " with CUDA " + CUDA_COMPILER_VERSION;
#else
	info = info + " without CUDA";
#endif
	return info;
}