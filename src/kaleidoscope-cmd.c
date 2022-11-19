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
	unsigned long long ctr, maxCtr = 1;
	double startTime, endTime;

	KaleidoscopeHandle handler;
	ImageData imgData, outData;

	// Parse inputs
	if (argc < 4)
	{
		fprintf(stderr, "Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N>\n");
		fprintf(stderr,
				"Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N> <Dim constant> <Scale factor>\n");
		return retval;
	}
	fprintf(stderr, "Start...\n");

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

	// Check inputs
	if (n <= 2)
	{
		fprintf(stderr, "n should be greater than 2");
		return EXIT_FAILURE;
	}
	if (scaleDown < 0.0 || scaleDown > 1.0)
	{
		fprintf(stderr, "Scale factor should be between 0.0 and 1.0");
		return EXIT_FAILURE;
	}

	// Process
	fprintf(stderr, "Reading %s ... ", path);
	if ((retval = readImage(path, &imgData)))
		return retval;
	fprintf(stderr, " %d\n", !retval);

	fprintf(stderr, "Initializing ... ");
	if (initImageData(&outData, imgData.width, imgData.height, imgData.nComponents))
		return EXIT_FAILURE;
	if ((retval = initKaleidoscope(&handler, n, imgData.width, imgData.height, scaleDown)))
		return retval;
	fprintf(stderr, " %d\n", !retval);

	fprintf(stderr, "Processing ...");
	startTime = (float)clock() / CLOCKS_PER_SEC;
	for (ctr = 0; ctr < maxCtr; ++ctr)
	{
		processKaleidoscope(&handler, k, &imgData, &outData);
		if (!benchmark)
			break;
	}
	endTime = (float)clock() / CLOCKS_PER_SEC;
	fprintf(stderr, " 1\n");

	if (benchmark)
		fprintf(stderr, "FPS %5.3f\n", 1 / ((endTime - startTime) / maxCtr));

	fprintf(stderr, "Saving %s... ", outPath);
	if ((retval = saveImage(outPath, &outData, TJPF_RGB, TJSAMP_444, 90)))
		return retval;
	fprintf(stderr, " %d\n", !retval);

	deInitImageData(&imgData);
	deInitImageData(&outData);
	deInitKaleidoscope(&handler);

	fprintf(stderr, "Done...\n");

	return retval;
}
