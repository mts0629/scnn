/**
 * @file test_data_runner.c
 * @brief test runner of data.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(data)
{
    RUN_TEST_CASE(data, fdata_alloc);

    RUN_TEST_CASE(data, fdata_copy);
}
