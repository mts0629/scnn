/**
 * @file test_data.c
 * @brief unit tests of data.h
 * 
 */
#include "data.h"
#if 0
#include "util.h"
#endif

#include "unity_fixture.h"

TEST_GROUP(data);

TEST_SETUP(data)
{}

TEST_TEAR_DOWN(data)
{}

TEST(data, fdata_alloc)
{
    float *f = fdata_alloc(10);
    TEST_ASSERT_NOT_NULL(f);

    float arr[10];
    for (int i = 0; i < 10; i++) {
        f[i] = i;
        arr[i] = i;
    }

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(arr, f, 10);

#if 0
    FREE_WITH_NULL(&f);
#endif
    free(f);
    f = NULL;

    TEST_ASSERT_NULL(f);
}

TEST(data, fdata_copy)
{
    float d1[10];
    for (int i = 0; i < 10; i++) {
        d1[i] = i;
    }

    float d2[10] = { 0 };

    fdata_copy(d1, 10, d2);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(d2, d1, 10);
}
