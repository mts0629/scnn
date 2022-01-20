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
    float *mat = mat_alloc(2, 3);
    TEST_ASSERT_NOT_NULL(mat);

    mat_free(&mat);
    TEST_ASSERT_NULL(mat);
}

TEST(mat, test_mat_alloc_invalid_sizes)
{
    TEST_ASSERT_NULL(mat_alloc(0, 1));

    TEST_ASSERT_NULL(mat_alloc(1, 0));
}

TEST(mat, test_mat_free_null)
{
    float *mat = NULL;

    mat_free(&mat);
    TEST_ASSERT_NULL(mat);
}

TEST(mat, test_mat_copy)
{
    float *dst = mat_alloc(2, 3);

    float src[2 * 3] = {
        0, 1, 2,
        3, 4, 5
    };

    float *ptr = mat_copy(src, 2, 3, dst);

    TEST_ASSERT_NOT_NULL(ptr);
    TEST_ASSERT_EQUAL_PTR(dst, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(src, dst, (2 * 3));

    mat_free(&dst);
}

TEST(mat, test_mat_copy_invalid_sizes)
{
    float *dst = mat_alloc(2, 3);

    float zeros[2 * 3] = { 0 };

    mat_copy(zeros, 2, 3, dst);

    float src[2 * 3] = {
        0, 1, 2,
        3, 4, 5
    };

    TEST_ASSERT_NULL(mat_copy(src, 0, 3, dst));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(zeros, dst, (2 * 3));

    TEST_ASSERT_NULL(mat_copy(src, 2, 0, dst));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(zeros, dst, (2 * 3));

    mat_free(&dst);
}

TEST(mat, test_mat_copy_null)
{
    float *dst = mat_alloc(2, 3);

    float dst_vals[2 * 3] = {
        0, 1, 2,
        3, 4, 5
    };

    mat_copy(dst_vals, 2, 3, dst);

    TEST_ASSERT_NULL(mat_copy(NULL, 2, 3, dst));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dst_vals, dst, (2 * 3));

    float zeros[2 * 3] = { 0 };

    TEST_ASSERT_NULL(mat_copy(zeros, 2, 3, NULL));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dst_vals, dst, (2 * 3));

    mat_free(&dst);
}

TEST(mat, test_mat_fill)
{
    float *mat = mat_alloc(2, 3);

    TEST_ASSERT_EQUAL_PTR(mat, mat_fill(mat, 2, 3, 1));

    TEST_ASSERT_EACH_EQUAL_FLOAT(1, mat, (2 * 3));

    mat_free(&mat);
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

    float c[3 * 2] = { 0 };

    float ans[3 * 2] = {
        -2, 0,
        2, 4,
        6, 8 
    };

    float *ptr = mat_add(a, b, c, 3, 2);

    TEST_ASSERT_EQUAL_PTR(c, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, c, (3 * 2));
}

TEST(mat, test_mat_add_accumulate)
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

    float ans[3 * 2] = {
        -2, 0,
        2, 4,
        6, 8 
    };

    float *ptr = mat_add(a, b, b, 3, 2);

    TEST_ASSERT_EQUAL_PTR(b, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, b, (3 * 2));
}

TEST(mat, test_mat_add_accumulate_inplace)
{
    float a[3 * 2] = {
        0, 1,
        2, 3,
        4, 5 
    };

    float ans[3 * 2] = {
        0, 2,
        4, 6,
        8, 10 
    };

    float *ptr = mat_add(a, a, a, 3, 2);

    TEST_ASSERT_EQUAL_PTR(a, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, a, (3 * 2));
}

TEST(mat, test_mat_add_invalid_sizes)
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

    float c[3 * 2] = { 0 };

    TEST_ASSERT_NULL(mat_add(a, b, c, 3, 0));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 2));

    TEST_ASSERT_NULL(mat_add(a, b, c, 0, 2));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 2));

    TEST_ASSERT_NULL(mat_add(a, b, c, 3, -1));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 2));

    TEST_ASSERT_NULL(mat_add(a, b, c, -1, 2));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 2));
}

TEST(mat, test_mat_add_null)
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

    float c[3 * 2] = { 0 };

    TEST_ASSERT_NULL(mat_add(NULL, b, c, 3, 2));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 2));

    TEST_ASSERT_NULL(mat_add(a, NULL, c, 3, 2));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 2));

    TEST_ASSERT_NULL(mat_add(a, b, NULL, 3, 2));
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

    float c[3 * 4] = { 0 };

    float ans[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    float *ptr = mat_mul(a, b, c, 3, 2, 4);

    TEST_ASSERT_EQUAL_PTR(c, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, c, (3 * 4));
}

TEST(mat, test_mat_mul_invalid_sizes)
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

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul(a, b, c, 0, 2, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, b, c, 3, 0, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, b, c, 3, 2, 0));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, b, c, -1, 2, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, b, c, 3, -1, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, b, c, 3, 2, -1));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
}

TEST(mat, test_mat_mul_null)
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

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul(NULL, b, c, 3, 2, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, NULL, c, 3, 2, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul(a, b, NULL, 3, 2, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
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

    float c[3 * 4] = { 0 };

    float ans[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    float *ptr = mat_mul_trans_a(a, b, c, 2, 3, 4);

    TEST_ASSERT_EQUAL_PTR(c, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_a_invalid_sizes)
{
    float a[2 * 3] = {
        0, 2, 4,
        1, 3, 5
    };

    float b[2 * 4] = {
        -2, -1, 0, 1,
        0, 1, 2, 3,
    };

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul_trans_a(a, b, c, 0, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_a(a, b, c, 2, 0, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_a(a, b, c, 2, 3, 0));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_a_null)
{
    float a[2 * 3] = {
        0, 2, 4,
        1, 3, 5
    };

    float b[2 * 4] = {
        -2, -1, 0, 1,
        0, 1, 2, 3,
    };

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul_trans_a(NULL, b, c, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_a(a, NULL, c, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_a(a, b, NULL, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
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

    float c[3 * 4] = { 0 };

    float ans[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    float *ptr = mat_mul_trans_b(a, b, c, 3, 2, 4);

    TEST_ASSERT_EQUAL_PTR(c, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_b_invalid_sizes)
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

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul_trans_b(a, b, c, 0, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_b(a, b, c, 2, 0, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_b(a, b, c, 2, 3, 0));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_b_null)
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

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul_trans_b(NULL, b, c, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_b(a, NULL, c, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_b(a, b, NULL, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
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

    float c[3 * 4] = { 0 };

    float ans[3 * 4] = {
        0, 1, 2, 3,
        -4, 1, 6, 11,
        -8, 1, 10, 19
    };

    float *ptr = mat_mul_trans_ab(a, b, c, 2, 3, 4);

    TEST_ASSERT_EQUAL_PTR(c, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_ab_invalid_sizes)
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

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul_trans_ab(a, b, c, 0, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_ab(a, b, c, 2, 0, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_ab(a, b, c, 2, 3, 0));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
}

TEST(mat, test_mat_mul_trans_ab_null)
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

    float c[3 * 4] = { 0 };

    TEST_ASSERT_NULL(mat_mul_trans_ab(NULL, b, c, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_ab(a, NULL, c, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));

    TEST_ASSERT_NULL(mat_mul_trans_ab(a, b, NULL, 2, 3, 4));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, c, (3 * 4));
}

TEST(mat, test_mat_mul_scalar)
{
    float a[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3
    };

    float k = -1.0;

    float b[3 * 2] = { 0 };

    float ans[3 * 2] = {
        2, 1,
        0, -1,
        -2, -3
    };

    float *ptr = mat_mul_scalar(a, b, 3, 2, k);

    TEST_ASSERT_EQUAL_PTR(b, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, b, (3 * 2));
}

TEST(mat, test_mat_mul_scalar_inplace)
{
    float a[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3
    };

    float k = -1.0;

    float ans[3 * 2] = {
        2, 1,
        0, -1,
        -2, -3
    };

    float *ptr = mat_mul_scalar(a, a, 3, 2, k);

    TEST_ASSERT_EQUAL_PTR(a, ptr);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, a, (3 * 2));
}

TEST(mat, test_mat_mul_scalar_invalid_sizes)
{
    float a[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3
    };

    float k = -1.0;

    float b[3 * 2] = { 0 };

    TEST_ASSERT_NULL(mat_mul_scalar(a, b, 0, 2, k));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, b, (3 * 2));

    TEST_ASSERT_NULL(mat_mul_scalar(a, b, 3, 0, k));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, b, (3 * 2));

    TEST_ASSERT_NULL(mat_mul_scalar(a, b, -1, 2, k));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, b, (3 * 2));

    TEST_ASSERT_NULL(mat_mul_scalar(a, b, 3, -1, k));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, b, (3 * 2));
}

TEST(mat, test_mat_mul_scalar_null)
{
    float a[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3
    };

    float k = -1.0;

    float b[3 * 2] = { 0 };

    TEST_ASSERT_NULL(mat_mul_scalar(NULL, b, 3, 2, k));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, b, (3 * 2));

    TEST_ASSERT_NULL(mat_mul_scalar(a, NULL, 3, 2, k));
    TEST_ASSERT_EACH_EQUAL_FLOAT(0, b, (3 * 2));
}
