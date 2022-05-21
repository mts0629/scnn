/**
 * @file test_net_runner.c
 * @brief 
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(net)
{
    RUN_TEST_CASE(net, net_alloc_and_free);

    RUN_TEST_CASE(net, net_create_and_free);

    RUN_TEST_CASE(net, net_create_over_size);

    RUN_TEST_CASE(net, net_append);

    RUN_TEST_CASE(net, net_append_null);

    RUN_TEST_CASE(net, net_init_layer_params);

    RUN_TEST_CASE(net, net_forward);

    RUN_TEST_CASE(net, net_backward);
}
