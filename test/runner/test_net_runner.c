/**
 * @file test_net_runner.c
 * @brief 
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(net)
{
    RUN_TEST_CASE(net, net_create_and_free);
}
