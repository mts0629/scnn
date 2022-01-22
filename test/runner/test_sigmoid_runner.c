/**
 * @file test_sigmoid_runner.c
 * @brief test runner of sigmoid.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(sigmoid)
{
    RUN_TEST_CASE(sigmoid, sigmoid_alloc_and_free);

    RUN_TEST_CASE(sigmoid, sigmoid_alloc_invalid_param);

    RUN_TEST_CASE(sigmoid, sigmoid_forward);

    RUN_TEST_CASE(sigmoid, sigmoid_backward);
}
