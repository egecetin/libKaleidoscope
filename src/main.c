#include "header.h"

int main(int argc, char *argv[])
{
    int retval = -1;
    printf("Start...\n");

    /*
    if(argc < 3)
    {
        printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path>");
        return retval;
    }

    ImageData imgData;
    printf("Reading %s... ", argv[1]);
    retval = readImage(argv[1], &imgData);
    printf(" %d\n", retval);

    printf("Dimming ... ");
    retval = dimBackground(&imgData, 0.5, nullptr);
    printf(" %d\n", retval);

    printf("Saving %s... ", argv[2]);
    retval = saveImage(argv[2], &imgData);
    printf(" %d\n", retval);

    printf("Done...\n");
    return retval;
    */

    // Only for debug
    char path[] = "/mnt/c/Users/egece/Pictures/deneme.jpg";
    char outPath[] = "/mnt/c/Users/egece/Pictures/out.jpg";

    ImageData imgData;
    printf("Reading %s... ", path);
    retval = readImage(path, &imgData);
    printf(" %d\n", retval);

    retval = dimBackground(&imgData, 0.5, nullptr);

    printf("Saving %s... ", outPath);
    retval = saveImage(outPath, &imgData);
    printf(" %d\n", retval);

    printf("Done...\n");
    return 0;
}