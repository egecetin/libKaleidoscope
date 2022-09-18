#include "kaleidoscope.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int retval = -1;

	// Parse inputs
	if (argc < 4)
	{
		printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N>\n");
		printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N> <Dim constant> <Scale factor>\n");
		return retval;
	}
	printf("Start...\n");

	char *path = argv[1];
	char *outPath = argv[2];
	int n = atoi(argv[3]);
	double k = 0.30;
	double scaleDown = 0.45;
	if (argc > 4)
	{
		k = atof(argv[4]);
		scaleDown = atof(argv[5]);
	}

	// Process
	ImageData imgData;
	printf("Reading %s... ", path);
	if ((retval = readImage(path, &imgData)))
		return retval;
	printf(" %d\n", !retval);

	printf("Applying effect... ");
	if ((retval = kaleidoscope(&imgData, n, k, scaleDown)))
		return retval;
	printf(" %d\n", !retval);

	printf("Saving %s... ", outPath);
	if ((retval = saveImage(outPath, &imgData)))
		return retval;
	printf(" %d\n", !retval);

	printf("Done...\n");

	return retval;
}