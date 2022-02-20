/**
 * @file test_data.c
 * @brief unit tests of data.h
 * 
 */
#include "data.h"
#include "util.h"
#include "random.h"

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

    FREE_WITH_NULL(&f);

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

TEST(data, fdata_rand_uniform)
{
    rand_seed(0);

    float m1[100];
    float m2[100];

    fdata_rand_uniform(m1, 100);
    fdata_rand_uniform(m2, 100);

    int cnt = 0;
    for (int i = 0; i < 100; i++) {
        if (m1[i] == m2[i]) {
            cnt++;
        }
        TEST_ASSERT((m1[i] >= 0) && (m1[i] <= 1));
        TEST_ASSERT((m2[i] >= 0) && (m2[i] <= 1));
    }

    TEST_ASSERT(cnt < 100);
}

TEST(data, fdata_rand_norm)
{
    rand_seed(0);

    float m1[100];
    float m2[100];

    fdata_rand_norm(m1, 100, 0, 1);
    fdata_rand_norm(m2, 100, 0, 1);

    int cnt = 0;
    for (int i = 0; i < 100; i++) {
        if (m1[i] == m2[i]) {
            cnt++;
        }
    }

    TEST_ASSERT(cnt < 100);
}
