#include <check.h>
#include <kaleidoscope.h>
#include <stdlib.h>

START_TEST(ImageDataTest)
{
	ImageData *imgPtr = NULL;
	int width = 1024, height = 768, nComponents = 3;

	// Init
	imgPtr = (ImageData *)malloc(sizeof(ImageData));
	ck_assert_ptr_nonnull(imgPtr);

	// Check
	ck_assert_int_eq(0, initImageData(imgPtr, width, height, nComponents));
	ck_assert_int_eq(width, imgPtr->width);
	ck_assert_int_eq(height, imgPtr->height);
	ck_assert_int_eq(nComponents, imgPtr->nComponents);
	ck_assert_ptr_nonnull(imgPtr->data);

	// De-init
	deInitImageData(imgPtr);
	ck_assert_ptr_null(imgPtr->data);

	free(imgPtr);
}
END_TEST

Suite *imageDataTestSuite(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("ImageDataTest");
	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, ImageDataTest);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = imageDataTestSuite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
