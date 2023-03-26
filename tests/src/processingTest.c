#include <check.h>
#include <kaleidoscope.h>
#include <stdio.h>
#include <stdlib.h>

START_TEST(Processing)
{
	double k = 0.30;
	double scaleDown = 0.45;
	int n = 6, width = 1935, height = 1088, nComponents = 3;
	FILE *fIn = NULL, *fExpected = NULL;

	KaleidoscopeHandle handler;
	unsigned char *inData, *outData, *expectedData;

	char pathIn[] = "../../tests/data/processing_1935x1088_InputData.bin";
	char pathExpected[] = "../../tests/data/processing_1935x1088_ExpectedData.bin";

	// Init
	fIn = fopen(pathIn, "rb");
	fExpected = fopen(pathExpected, "rb");

	ck_assert_ptr_nonnull(fIn);
	ck_assert_ptr_nonnull(fExpected);

	inData = (unsigned char *)calloc(width * height * nComponents * sizeof(unsigned char), 1);
	outData = (unsigned char *)calloc(width * height * nComponents * sizeof(unsigned char), 1);
	expectedData = (unsigned char *)calloc(width * height * nComponents * sizeof(unsigned char), 1);

	ck_assert_ptr_nonnull(inData);
	ck_assert_ptr_nonnull(outData);
	ck_assert_ptr_nonnull(expectedData);

	// Read data
	ck_assert_int_lt(0, fread(inData, width * height * nComponents, 1, fIn));
	ck_assert_int_lt(0, fread(expectedData, width * height * nComponents, 1, fExpected));

	// Process
	ck_assert_int_eq(0, initKaleidoscope(&handler, k, n, width, height, nComponents, scaleDown));
	processKaleidoscope(&handler, inData, outData);

	// Check
	ck_assert_mem_eq(expectedData, outData, width * height * nComponents);

	// De-init
	fclose(fIn);
	fclose(fExpected);

	free(inData);
	free(outData);
	free(expectedData);

	deInitKaleidoscope(&handler);
}
END_TEST

Suite *processingTestSuite(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("ProcessingTest");
	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, Processing);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = processingTestSuite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
