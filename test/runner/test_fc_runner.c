/**
 * @file test_fc_runner.c
 * @brief test runner of fc.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(fc)
{
    RUN_TEST_CASE(fc, fc_alloc_and_free);

    RUN_TEST_CASE(fc, fc_forward);
}