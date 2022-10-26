#include <check.h>
#include <kaleidoscope.h>
#include <stdlib.h>

START_TEST(ImageDataTest)
{
    ImageData *imgPtr = NULL;
    int width = 1024, height = 768, nComponents = 3;

    imgPtr = (ImageData*)malloc(sizeof(ImageData));
    ck_assert_ptr_nonnull(imgPtr);

    ck_assert_int_eq(0, initImageData(imgPtr, width, height, nComponents));
    ck_assert_int_eq(width, imgPtr->width);
    ck_assert_int_eq(height, imgPtr->height);
    ck_assert_int_eq(nComponents, imgPtr->nComponents);
    ck_assert_ptr_nonnull(imgPtr->data);

    deInitImageData(imgPtr);
    ck_assert_ptr_null(imgPtr->data);

    free(imgPtr);
}
END_TEST

int main(void) { return 0; }