/**
 * @file test_fc_runner.c
 * @brief test runner of fc.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(fc)
{
    RUN_TEST_CASE(fc, fc_layer_and_free);

    RUN_TEST_CASE(fc, fc_layer_invalid_param);

    RUN_TEST_CASE(fc, fc_forward);

    RUN_TEST_CASE(fc, fc_backward);

    RUN_TEST_CASE(fc, fc_init);

    RUN_TEST_CASE(fc, fc_update);
}
