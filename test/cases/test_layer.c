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
    Layer *layer = layer_alloc();

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL(-1, layer->id);

    TEST_ASSERT_NULL(layer->x);

    TEST_ASSERT_NULL(layer->y);

    TEST_ASSERT_NULL(layer->w);
    TEST_ASSERT_NULL(layer->b);

    TEST_ASSERT_NULL(layer->dx);
    TEST_ASSERT_NULL(layer->dw);
    TEST_ASSERT_NULL(layer->db);

    TEST_ASSERT_EQUAL_INT(-1, layer->prev_id);
    TEST_ASSERT_EQUAL_INT(-1, layer->next_id);

    TEST_ASSERT_NULL(layer->forward);
    TEST_ASSERT_NULL(layer->backward);

    TEST_ASSERT_NOT_NULL(layer->update);

    const float *ptr_x = layer->x;

    float *ptr_y = layer->y;

    float *ptr_w = layer->w;
    float *ptr_b = layer->b;

    float *ptr_dx = layer->dx;
    float *ptr_dw = layer->dw;
    float *ptr_db = layer->db;

    layer_free(&layer);

    TEST_ASSERT_NULL(layer);

    TEST_ASSERT_NULL(ptr_x);

    TEST_ASSERT_NULL(ptr_y);

    TEST_ASSERT_NULL(ptr_w);
    TEST_ASSERT_NULL(ptr_b);

    TEST_ASSERT_NULL(ptr_dx);
    TEST_ASSERT_NULL(ptr_dw);
    TEST_ASSERT_NULL(ptr_db);
}
