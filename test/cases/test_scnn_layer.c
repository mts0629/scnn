/**
 * @file test_scnn_layer.c
 * @brief Unit tests of scnn_layer.c
 * 
 */
#include "scnn_layer.h"

// Private header, include to verify private members
#include "impl/scnn_layer_impl.h"

#include "unity.h"

#include "mock_scnn_mat.h"

scnn_layer *layer;

void setUp(void)
{
    layer = NULL;
}

void tearDown(void)
{}

void test_allocate_and_free(void)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 28, 28 } };
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL(SCNN_LAYER_NONE, layer->params.type);
    TEST_ASSERT_EQUAL_INT_ARRAY(params.in_shape, layer->params.in_shape, 4);

    TEST_ASSERT_EQUAL(0, layer->params.id);
    TEST_ASSERT_EQUAL(0, layer->params.prev_id);
    TEST_ASSERT_EQUAL(0, layer->params.next_id);

    TEST_ASSERT_NULL(layer->x);
    TEST_ASSERT_NULL(layer->y);
    TEST_ASSERT_NULL(layer->w);
    TEST_ASSERT_NULL(layer->b);
    TEST_ASSERT_NULL(layer->dx);
    TEST_ASSERT_NULL(layer->dw);
    TEST_ASSERT_NULL(layer->db);

    TEST_ASSERT_EQUAL(NULL, layer->init);
    TEST_ASSERT_EQUAL(NULL, layer->forward);
    TEST_ASSERT_EQUAL(NULL, layer->backward);

    scnn_mat_free_Expect(&layer->x);
    scnn_mat_free_Expect(&layer->y);
    scnn_mat_free_Expect(&layer->w);
    scnn_mat_free_Expect(&layer->b);
    scnn_mat_free_Expect(&layer->dx);
    scnn_mat_free_Expect(&layer->dw);
    scnn_mat_free_Expect(&layer->db);
    scnn_layer_free(&layer);
    TEST_ASSERT_NULL(layer);
}

void test_free_pointer_to_NULL(void)
{
    scnn_layer_free(&layer);
}

void test_free_NULL(void)
{
    scnn_layer_free(NULL);
}
