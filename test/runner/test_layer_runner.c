/**
 * @file test_layer_runner.c
 * @brief 
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(layer)
{
    RUN_TEST_CASE(layer, layer_alloc_and_free);
}
