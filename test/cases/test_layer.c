/**
 * @file test_layer.c
 * @brief 
 * 
 */
#include "layer.h"

#include "unity_fixture.h"

TEST_GROUP(layer);

TEST_SETUP(layer)
{}

TEST_TEAR_DOWN(layer)
{}

TEST(layer, layer_alloc_and_free)
{
    LayerParameter param = { .name = "layer", .in = 1, .in_h = 1, .in_w = 1, .out = 2 };
    Layer *layer = layer_alloc(param);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL_CHAR_ARRAY(param.name, layer->name, sizeof(param.name));

    TEST_ASSERT_NULL(layer->x);
    TEST_ASSERT_EQUAL_INT32(param.in, layer->in);
    TEST_ASSERT_EQUAL_INT32(param.in_h, layer->in_h);
    TEST_ASSERT_EQUAL_INT32(param.in_w, layer->in_w);

    TEST_ASSERT_NULL(layer->y);
    TEST_ASSERT_EQUAL_INT32(param.out, layer->out);

    TEST_ASSERT_NULL(layer->prev);
    TEST_ASSERT_NULL(layer->next);

    TEST_ASSERT_NULL(layer->forward);

    layer_free(&layer);

    TEST_ASSERT_NULL(layer);
}
