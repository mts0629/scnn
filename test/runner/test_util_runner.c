/**
 * @file test_util_runner.c
 * @brief test runner of util.h
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(util)
{
    RUN_TEST_CASE(util, free_with_null);
}
