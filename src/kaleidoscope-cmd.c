#include "jpeg-utils/jpeg-utils.h"
#include "kaleidoscope.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
	char *path = NULL, *outPath = NULL;
	int n = 6, retval = EXIT_FAILURE, benchmark = 0;
	double k = 0.30;
	double scaleDown = 0.45;
	unsigned long long ctr, maxCtr = 0;
	double startTime, endTime;

	KaleidoscopeHandle handler;
	ImageData imgData, outData;

	// Parse inputs
	if (argc < 4)
	{
		printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N>\n");
		printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N> <Dim constant> <Scale factor>\n");
		return retval;
	}
	printf("Start...\n");

	path = argv[1];
	outPath = argv[2];
	n = atoi(argv[3]);
	if (argc == 5)
	{
		benchmark = 1;
		maxCtr = atoll(argv[4]);
	}
	if (argc == 6)
	{
		k = atof(argv[4]);
		scaleDown = atof(argv[5]);
	}

	// Process
	printf("Reading %s ... ", path);
	if ((retval = readImage(path, &imgData)))
		return retval;
	printf(" %d\n", !retval);

	printf("Initializing ... ");
	if (initImageData(&outData, imgData.width, imgData.height, imgData.nComponents))
		return EXIT_FAILURE;
	if ((retval = initKaleidoscope(&handler, n, imgData.width, imgData.height, scaleDown)))
		return retval;
	printf(" %d\n", !retval);

	printf("Processing ...");
	startTime = (float)clock() / CLOCKS_PER_SEC;
	for (ctr = 0; ctr < maxCtr; ++ctr)
	{
		processKaleidoscope(&handler, k, &imgData, &outData);
		if (!benchmark)
			break;
	}
	endTime = (float)clock() / CLOCKS_PER_SEC;
	printf(" 1\n");

	if (benchmark)
		printf("FPS %5.3f\n", 1 / ((endTime - startTime) / maxCtr));

	printf("Saving %s... ", outPath);
	if ((retval = saveImage(outPath, &outData, TJPF_RGB, TJSAMP_444, 90)))
		return retval;
	printf(" %d\n", !retval);

	deInitImageData(&imgData);
	deInitImageData(&outData);
	deInitKaleidoscope(&handler);

	printf("Done...\n");

	return retval;
}
