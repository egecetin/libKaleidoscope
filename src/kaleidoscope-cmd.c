#include "jpeg-utils/jpeg-utils.h"
#include "kaleidoscope.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	char *path = NULL, *outPath = NULL;
	int n = 6, retval = EXIT_FAILURE;
	double k = 0.30;
	double scaleDown = 0.45;

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

	if (argc > 4)
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
	processKaleidoscope(&handler, k, &imgData, &outData);
	printf(" 1\n");

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