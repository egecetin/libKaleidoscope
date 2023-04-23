#include <check.h>
#include <kaleidoscope.h>
#include <stdlib.h>

START_TEST(VersionTest)
{
	int major = -1, minor = -1, patch = -1;
	getKaleidoscopeVersion(NULL, &minor, &patch);
	ck_assert_int_eq(major, -1);
	ck_assert_int_eq(minor, -1);
	ck_assert_int_eq(patch, -1);

	getKaleidoscopeVersion(&major, NULL, &patch);
	ck_assert_int_eq(major, -1);
	ck_assert_int_eq(minor, -1);
	ck_assert_int_eq(patch, -1);

	getKaleidoscopeVersion(&major, &minor, NULL);
	ck_assert_int_eq(major, -1);
	ck_assert_int_eq(minor, -1);
	ck_assert_int_eq(patch, -1);

	getKaleidoscopeVersion(&major, &minor, &patch);
	ck_assert_int_ne(major, -1);
	ck_assert_int_ne(minor, -1);
	ck_assert_int_ne(patch, -1);

	// Check version strings
	ck_assert_ptr_nonnull(getKaleidoscopeVersionString());
	ck_assert_ptr_nonnull(getKaleidoscopeLibraryInfo());
}
END_TEST

Suite *versionTestSuite(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("VersionTest");
	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, VersionTest);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = versionTestSuite();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
