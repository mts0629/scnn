/**
 * @file test_mat_runner.c
 * @brief test runner of mat.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(mat)
{
    RUN_TEST_CASE(mat, test_mat_alloc_and_free);
    RUN_TEST_CASE(mat, test_mat_alloc_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_free_null);

    RUN_TEST_CASE(mat, test_mat_copy);
    RUN_TEST_CASE(mat, test_mat_copy_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_copy_null);

    RUN_TEST_CASE(mat, test_mat_add);
    RUN_TEST_CASE(mat, test_mat_add_accumulate);
    RUN_TEST_CASE(mat, test_mat_add_accumulate_inplace);
    RUN_TEST_CASE(mat, test_mat_add_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_add_null);

    RUN_TEST_CASE(mat, test_mat_mul);
    RUN_TEST_CASE(mat, test_mat_mul_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_mul_null);

    RUN_TEST_CASE(mat, test_mat_mul_trans_a);
    RUN_TEST_CASE(mat, test_mat_mul_trans_a_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_mul_trans_a_null);

    RUN_TEST_CASE(mat, test_mat_mul_trans_b);
    RUN_TEST_CASE(mat, test_mat_mul_trans_b_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_mul_trans_b_null);

    RUN_TEST_CASE(mat, test_mat_mul_trans_ab);
    RUN_TEST_CASE(mat, test_mat_mul_trans_ab_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_mul_trans_ab_null);

    RUN_TEST_CASE(mat, test_mat_mul_scalar);
    RUN_TEST_CASE(mat, test_mat_mul_scalar_inplace);
    RUN_TEST_CASE(mat, test_mat_mul_scalar_invalid_sizes);
    RUN_TEST_CASE(mat, test_mat_mul_scalar_null);
}
