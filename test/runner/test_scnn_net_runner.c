/**
 * @file test_scnn_net_runner.c
 * @brief Test runner of scnn_net.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_net)
{
    RUN_TEST_CASE(scnn_net, allocate);

    RUN_TEST_CASE(scnn_net, free_with_NULL);

    RUN_TEST_CASE(scnn_net, append_layer);
    RUN_TEST_CASE(scnn_net, append_2layers);
    RUN_TEST_CASE(scnn_net, append_3layers);

    RUN_TEST_CASE(scnn_net, cannot_append_if_net_is_NULL);
    RUN_TEST_CASE(scnn_net, cannot_append_if_layer_is_NULL);
    RUN_TEST_CASE(scnn_net, cannot_append_if_over_max_size);

    RUN_TEST_CASE(scnn_net, init_layer);
    RUN_TEST_CASE(scnn_net, init_2layers);
    RUN_TEST_CASE(scnn_net, init_3layers);

    //RUN_TEST_CASE(scnn_net, forward);
    //RUN_TEST_CASE(scnn_net, forward_net_is_NULL);
    //RUN_TEST_CASE(scnn_net, forward_x_is_NULL);
    //RUN_TEST_CASE(scnn_net, backward);
    //RUN_TEST_CASE(scnn_net, backward_net_is_NULL);
    //RUN_TEST_CASE(scnn_net, backward_t_is_NULL);
}
