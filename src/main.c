#include "header.h"

int main(int argc, char *argv[])
{
    int retval = -1;

    // Parse inputs
    if (argc < 4)
    {
        printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N>\n");
        return retval;
    }
    printf("Start...\n");

    char *path = argv[1];
    char *outPath = argv[2];
    int n = atoi(argv[3]);

    // Process
    ImageData imgData;
    printf("Reading %s... ", path);
    retval = readImage(path, &imgData);
    printf(" %d\n", !retval);

    printf("Applying effect... ");
    retval = kaleidoscope(&imgData, n, 0.30, 0.45);
    printf(" %d\n", !retval);

    printf("Saving %s... ", outPath);
    retval = saveImage(outPath, &imgData);
    printf(" %d\n", !retval);

    printf("Done...\n");

    return retval;
}