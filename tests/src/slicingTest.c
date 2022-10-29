#include <check.h>
#include <kaleidoscope.h>
#include <readTestData.h>
#include <stdlib.h>

START_TEST(SlicingTest)
{
	int width = 1935, height = 1088, n = 6;
	double scaleDown = 0.45;
	TransformationInfo *inPtr = NULL, *outPtr = NULL, *expectedOutPtr = NULL;
	char testDataInPath[] = "../../tests/data/slicing_1935x1088_InputData.bin";
	char testDataExpectedPath[] = "../../tests/data/slicing_1935x1088_ExpectedData.bin";

	// Init
	inPtr = (TransformationInfo *)calloc(width * height, sizeof(TransformationInfo));
	expectedOutPtr = (TransformationInfo *)calloc(width * height, sizeof(TransformationInfo));

	ck_assert_ptr_nonnull(inPtr);
	ck_assert_ptr_nonnull(expectedOutPtr);

	// Read data
	ck_assert_int_eq(width * height, readTransformInfo(testDataInPath, inPtr));
	ck_assert_int_eq(width * height, readTransformInfo(testDataExpectedPath, expectedOutPtr));

	// Check
	sliceTriangle(inPtr, width, height, n, scaleDown);

	ck_assert_mem_eq(inPtr, expectedOutPtr, width * height * sizeof(TransformationInfo));

	// De-init
	free(inPtr);
	free(expectedOutPtr);
}
END_TEST

Suite *slicingTestSuite(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("SlicingTest");
	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, SlicingTest);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = slicingTestSuite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
