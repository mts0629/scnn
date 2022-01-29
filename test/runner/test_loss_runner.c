/**
 * @file test_loss_runner.c
 * @brief test runner of loss.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(loss)
{
    RUN_TEST_CASE(loss, mean_squared_error);
}
