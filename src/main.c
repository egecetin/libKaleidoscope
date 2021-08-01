#include "header.h"

/* TODO
    After scale down one point have more than one data (Average degrades performance)
    Interpolation after rotation
*/

int main(int argc, char *argv[])
{
    int retval = -1;
    printf("Start...\n");

    /*
    if(argc < 4)
    {
        printf("Usage ./kaleidoscope <Input Image Path> <Output Image Path> <N>");
        return retval;
    }

    char *path = argv[1];
    char *outPath = argv[2];
    int n = atoi(argv[3]);
    */

    // Only for debug
    char path[] = "/mnt/c/Users/egece/Pictures/deneme.jpg";
    char outPath[] = "/mnt/c/Users/egece/Pictures/out.jpg";
    int n = 11;

    ImageData imgData;
    printf("Reading %s... ", path);
    retval = readImage(path, &imgData);
    printf(" %d\n", !retval);

    printf("Applying effect... ");
    retval = kaleidoscope(&imgData, n, 0.35, 0.35);
    printf(" %d\n", !retval);

    printf("Saving %s... ", outPath);
    retval = saveImage(outPath, &imgData);
    printf(" %d\n", !retval);

    printf("Done...\n");

    return retval;
}