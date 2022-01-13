/**
 * @file test_mat.c
 * @brief unit tests of mat.c
 * 
 */
#include "mat.h"

#include "unity_fixture.h"

TEST_GROUP(mat);

TEST_SETUP(mat)
{}

TEST_TEAR_DOWN(mat)
{}

TEST(mat, test_mat_alloc_and_free)
{
    float *a = mat_alloc(2, 3);

    TEST_ASSERT_NOT_NULL(a);

    mat_free(&a);

    TEST_ASSERT_NULL(a);
}

TEST(mat, test_mat_alloc_invalid_rows)
{
    float *a = mat_alloc(0, 3);

    TEST_ASSERT_NULL(a);
}

TEST(mat, test_mat_alloc_invalid_columns)
{
    float *a = mat_alloc(2, 0);

    TEST_ASSERT_NULL(a);
}

TEST(mat, test_mat_free_null)
{
    float *a = mat_alloc(0, 0);

    mat_free(&a);

    TEST_ASSERT_NULL(a);
}

TEST(mat, test_mat_copy)
{
    float *a = mat_alloc(2, 3);

    float b[2 * 3] = { 0, 1, 2, 3, 4, 5 };

    mat_copy(b, 2, 3, a);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(a, b, (2 * 3));

    mat_free(&a);
}

TEST(mat, test_mat_add)
{
    float a[3 * 2] = {
        0, 1,
        2, 3,
        4, 5 
    };

    float b[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3 
    };

    float c[3 * 2];

    float y[3 * 2] = {
        -2, 0,
        2, 4,
        6, 8 
    };

    mat_add(a, b, c, 3, 2);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, c, (3 * 2));
}

TEST(mat, test_mat_mul)
{
    float a[3 * 2] = {
        0, 1,
        2, 3,
        4, 5 
    };

    float b[2 * 4] = {
        -2, -1, 0, 1,
        0, 1, 2, 3,
    };

    float c[3 * 4];

    float y[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    mat_mul(a, b, c, 3, 2, 4);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, c, (3 * 4));
}

TEST(mat, test_mat_mul_scalar)
{
    float a[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3
    };

    float k = -1.0;

    float b[3 * 2];

    float y[3 * 2] = {
        2, 1,
        0, -1,
        -2, -3
    };

    mat_mul_scalar(a, b, 3, 2, k);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, b, (3 * 2));
}

TEST(mat, test_mat_mul_trans_a)
{
    float a[2 * 3] = {
        0, 2, 4,
        1, 3, 5
    };

    float b[2 * 4] = {
        -2, -1, 0, 1,
        0, 1, 2, 3,
    };

    float c[3 * 4];

    float y[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    mat_mul_trans_a(a, b, c, 2, 3, 4);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_b)
{
    float a[3 * 2] = {
        0, 1,
        2, 3,
        4, 5 
    };

    float b[4 * 2] = {
        -2, 0,
        -1, 1,
        0, 2,
        1, 3
    };

    float c[3 * 4];

    float y[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    mat_mul_trans_b(a, b, c, 3, 2, 4);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_ab)
{
    float a[2 * 3] = {
        0, 2, 4,
        1, 3, 5
    };

    float b[4 * 2] = {
        -2, 0,
        -1, 1,
        0, 2,
        1, 3
    };

    float c[3 * 4];

    float y[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    mat_mul_trans_ab(a, b, c, 2, 3, 4);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, c, (3 * 4));
}
