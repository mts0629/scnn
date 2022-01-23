/**
 * @file test_softmax_runner.c
 * @brief test runner of softmax.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(softmax)
{
    RUN_TEST_CASE(softmax, softmax_alloc_and_free);

    RUN_TEST_CASE(softmax, softmax_alloc_invalid_param);

    RUN_TEST_CASE(softmax, softmax_forward);

    RUN_TEST_CASE(softmax, softmax_backward);
}
