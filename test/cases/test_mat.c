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
