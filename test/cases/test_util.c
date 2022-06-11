/**
 * @file test_util.c
 * @brief unit tests of util.h
 * 
 */
#include <stdlib.h>

#include "util.h"

#include "unity_fixture.h"

TEST_GROUP(util);

TEST_SETUP(util)
{}

TEST_TEAR_DOWN(util)
{}

TEST(util, free_with_null)
{
    int *a = malloc(sizeof(int) * 10);

    TEST_ASSERT_NOT_NULL(a);

    FREE_WITH_NULL(&a);

    TEST_ASSERT_NULL(a);
}
