#include "header.h"

int main()
{
    printf("Start...\n");

    char path[] = "/mnt/c/Users/egece/Pictures/deneme.jpg";
    char outPath[] = "/mnt/c/Users/egece/Pictures/out.jpg";

    ImageData imgData;
    printf("Reading...\n");
    int retval = readImage(path, &imgData);
    printf("%d\n", retval);

    printf("Saving...\n");
    retval = saveImage(outPath, &imgData);
    printf("%d\n", retval);

    return 0;
}