#include <check.h>
#include <kaleidoscope.h>
#include <readTestData.h>
#include <stdlib.h>

START_TEST(Rotation)
{
	int width = 1935, height = 1088;
	TransformationInfo *inPtr = NULL, *outPtr = NULL, *expectedOutPtr = NULL;
	char testDataInPath[] = "../../tests/data/rotation_1935x1088_InputData.bin";
	char testDataExpectedPath[] = "../../tests/data/rotation_1935x1088_ExpectedData.bin";

	// Init
	inPtr = (TransformationInfo *)calloc(width * height, sizeof(TransformationInfo));
	outPtr = (TransformationInfo *)calloc(width * height, sizeof(TransformationInfo));
	expectedOutPtr = (TransformationInfo *)calloc(width * height, sizeof(TransformationInfo));

	ck_assert_ptr_nonnull(inPtr);
	ck_assert_ptr_nonnull(outPtr);
	ck_assert_ptr_nonnull(expectedOutPtr);

	// Read data
	ck_assert_int_eq(width * height, readTransformInfo(testDataInPath, inPtr));
	ck_assert_int_eq(width * height, readTransformInfo(testDataExpectedPath, expectedOutPtr));

	// Check
	rotatePoints(outPtr, inPtr, width, height, 0);
	rotatePoints(outPtr, inPtr, width, height, 60);
	rotatePoints(outPtr, inPtr, width, height, 120);
	rotatePoints(outPtr, inPtr, width, height, 180);
	rotatePoints(outPtr, inPtr, width, height, 240);
	rotatePoints(outPtr, inPtr, width, height, 300);

	ck_assert_mem_eq(outPtr, expectedOutPtr, width * height * sizeof(TransformationInfo));

	// De-init
	free(inPtr);
	free(outPtr);
	free(expectedOutPtr);
}
END_TEST

Suite *rotationTestSuite(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("RotationTest");
	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, Rotation);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = rotationTestSuite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
