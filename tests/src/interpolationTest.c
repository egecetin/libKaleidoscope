#include <check.h>
#include <c/kaleidoscope.h>
#include <readTestData.h>
#include <stdlib.h>

START_TEST(InterpolationTest)
{
	int width = 1935, height = 1088;
	struct TransformationInfo_t *inPtr = NULL, *outPtr = NULL, *expectedOutPtr = NULL;
	char testDataInPath[] = "../../tests/data/interpolation_1935x1088_InputData.bin";
	char testDataExpectedPath[] = "../../tests/data/interpolation_1935x1088_ExpectedData.bin";

	// Init
	inPtr = (struct TransformationInfo_t *)calloc(width * height, sizeof(struct TransformationInfo_t));
	outPtr = (struct TransformationInfo_t *)calloc(width * height, sizeof(struct TransformationInfo_t));
	expectedOutPtr = (struct TransformationInfo_t *)calloc(width * height, sizeof(struct TransformationInfo_t));

	ck_assert_ptr_nonnull(inPtr);
	ck_assert_ptr_nonnull(outPtr);
	ck_assert_ptr_nonnull(expectedOutPtr);

	// Read data
	ck_assert_int_eq(width * height, readTransformInfo(testDataInPath, inPtr));
	ck_assert_int_eq(width * height, readTransformInfo(testDataExpectedPath, expectedOutPtr));

	// Check
	interpolate(outPtr, inPtr, width, height);

	ck_assert_mem_eq(outPtr, expectedOutPtr, width * height * sizeof(struct TransformationInfo_t));

	// De-init
	free(inPtr);
	free(outPtr);
	free(expectedOutPtr);
}
END_TEST

Suite *interpolationTestSuite(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("InterpolationTest");
	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, InterpolationTest);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = interpolationTestSuite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
