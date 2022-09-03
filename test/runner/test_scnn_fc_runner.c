/**
 * @file test_scnn_fc_runner.c
 * @brief Test runner of scnn_fc.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_fc)
{
    RUN_TEST_CASE(scnn_fc, alloc_and_free);

    RUN_TEST_CASE(scnn_fc, alloc_fail_invalid_param_in);
    RUN_TEST_CASE(scnn_fc, alloc_fail_invalid_param_out);

    RUN_TEST_CASE(scnn_fc, set_size);

    RUN_TEST_CASE(scnn_fc, set_size_fail_invalid_n);
    RUN_TEST_CASE(scnn_fc, set_size_fail_invalid_c);
    RUN_TEST_CASE(scnn_fc, set_size_fail_invalid_h);
    RUN_TEST_CASE(scnn_fc, set_size_fail_invalid_w);
    RUN_TEST_CASE(scnn_fc, set_size_fail_invalid_in_size);

    RUN_TEST_CASE(scnn_fc, forward);
    RUN_TEST_CASE(scnn_fc, forward_fail_x_is_null);
    RUN_TEST_CASE(scnn_fc, forward_fail_layer_is_null);

    RUN_TEST_CASE(scnn_fc, backward);
    RUN_TEST_CASE(scnn_fc, backward_fail_dy_is_null);
    RUN_TEST_CASE(scnn_fc, backward_fail_layer_is_null);
}
