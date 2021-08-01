#include "header.h"

/* TODO
    After scale down one point have more than one data (Average values?)
    Interpolation after rotation
*/

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
    */

    // Only for debug
    char path[] = "/mnt/c/Users/egece/Pictures/deneme.jpg";
    char outPath[] = "/mnt/c/Users/egece/Pictures/out.jpg";

    ImageData imgData;
    printf("Reading %s... ", path);
    retval = readImage(path, &imgData);
    printf(" %d\n", !retval);

    printf("Applying effect... ");
    retval = kaleidoscope(&imgData, 11, 0.35, 0.45);
    printf(" %d\n", !retval);

    printf("Saving %s... ", outPath);
    retval = saveImage(outPath, &imgData);
    printf(" %d\n", !retval);

    printf("Done...\n");

    return 0;
}