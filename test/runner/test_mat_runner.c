/**
 * @file test_mat_runner.c
 * @brief test runner of mat.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(mat)
{
    RUN_TEST_CASE(mat, test_mat_add);

    RUN_TEST_CASE(mat, test_mat_mul);
    RUN_TEST_CASE(mat, test_mat_mul_trans_a);
    RUN_TEST_CASE(mat, test_mat_mul_trans_b);
    RUN_TEST_CASE(mat, test_mat_mul_trans_ab);

    RUN_TEST_CASE(mat, test_mat_mul_scalar);
}
