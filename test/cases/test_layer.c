/**
 * @file test_layer.c
 * @brief unit tests of layer.c
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
    LayerParameter param = { .name = "layer" };
    Layer *layer = layer_alloc(param);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL_CHAR_ARRAY(param.name, layer->name, sizeof(param.name));

    TEST_ASSERT_NULL(layer->x);

    TEST_ASSERT_NULL(layer->y);

    TEST_ASSERT_NULL(layer->w);

    TEST_ASSERT_NULL(layer->b);

    TEST_ASSERT_NULL(layer->prev);
    TEST_ASSERT_NULL(layer->next);

    TEST_ASSERT_NULL(layer->forward);

    layer_free(&layer);

    TEST_ASSERT_NULL(layer);
}
