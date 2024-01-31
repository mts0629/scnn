/**
 * @file test_scnn_layer.c
 * @brief Unit tests of scnn_layer.c
 * 
 */
#include "scnn_layer.h"

#include <string.h>

#include "unity.h"

#include "scnn_blas.h"
#include "mock_scnn_fc.h"

static scnn_layer_params params = {
    SCNN_LAYER_FC, .in = 3 * 28 * 28, .out = 100,
};

static scnn_layer *layer;
static scnn_layer *layer_next;

static scnn_dtype x[3 * 28 * 28];
static scnn_dtype dy[100];

void setUp(void)
{
    layer = NULL;
    layer_next = NULL;
}

void tearDown(void)
{
    scnn_layer_free(&layer);
}

void test_allocate_and_free(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL(params.type, layer->params.type);
    TEST_ASSERT_EQUAL_INT(params.in, layer->params.in);
    TEST_ASSERT_EQUAL(params.out, layer->params.out);

    TEST_ASSERT_NULL(layer->x);
    TEST_ASSERT_NULL(layer->y);
    TEST_ASSERT_NULL(layer->w);
    TEST_ASSERT_NULL(layer->b);
    TEST_ASSERT_NULL(layer->dx);
    TEST_ASSERT_NULL(layer->dw);
    TEST_ASSERT_NULL(layer->db);

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

void test_init(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_EQUAL_PTR(layer, scnn_layer_init(layer));
    TEST_ASSERT_NOT_NULL(layer->x);
    TEST_ASSERT_NOT_NULL(layer->y);
    TEST_ASSERT_NOT_NULL(layer->w);
    TEST_ASSERT_NOT_NULL(layer->b);
    TEST_ASSERT_NOT_NULL(layer->dx);
    TEST_ASSERT_NOT_NULL(layer->dw);
    TEST_ASSERT_NOT_NULL(layer->db);

    scnn_layer_free(&layer);
}

void test_init_fail_if_layer_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_layer_init(NULL));
}

void test_connect(void)
{
    layer = scnn_layer_alloc(
        (scnn_layer_params){
            SCNN_LAYER_FC, .in =  3 * 28 * 28, .out = 100,
        }
    );
    layer_next = scnn_layer_alloc(
        (scnn_layer_params){
            SCNN_LAYER_FC, .out = 10,
        }
    );

    scnn_layer_connect(layer, layer_next);

    TEST_ASSERT_EQUAL_INT(100, layer_next->params.in);

    scnn_layer_free(&layer);
    scnn_layer_free(&layer_next);
}

void test_forward(void)
{
    layer = scnn_layer_alloc(
        (scnn_layer_params){
            SCNN_LAYER_FC, .in=2, .out=3,
        }
    );

    scnn_layer_init(layer);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    memcpy(layer->w, w, sizeof(w));

    scnn_dtype b[] = {
        1, 1, 1
    };
    memcpy(layer->b, b, sizeof(b));

    scnn_dtype _x[] = {
        1, 1
    };

    scnn_dtype answer[] = {
        4, 6, 8
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, scnn_layer_forward(layer, _x), 3);

    scnn_layer_free(&layer);
}

void test_forward_fail_if_layer_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_layer_forward(NULL, x));
}

void test_forward_fail_if_x_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NULL(scnn_layer_forward(layer, NULL));

    scnn_layer_free(&layer);
}

void test_backward(void)
{
    layer = scnn_layer_alloc(
        (scnn_layer_params){
            SCNN_LAYER_FC, .in = 2, .out = 3
        }
    );

    scnn_layer_init(layer);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    memcpy(layer->w, w, sizeof(w));

    scnn_dtype b[] = {
        1, 1, 1
    };
    memcpy(layer->b, b, sizeof(b));

    scnn_dtype _x[] = {
        1, 2
    };

    scnn_dtype _dy[] = {
        8, 12, 16
    };

    scnn_layer_forward(layer, _x);

    scnn_dtype _dx[] = {
        44, 152
    };

    scnn_dtype _dw[] = {
        8,  12, 16,
        16, 24, 32
    };

    scnn_dtype _db[] = {
        8, 12, 16
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_dx, scnn_layer_backward(layer, _dy), 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_dw, layer->dw, 2 * 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_db, layer->db, 3);

    scnn_layer_free(&layer);
}

void test_backward_fail_if_layer_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_layer_backward(NULL, dy));
}

void test_backward_fail_if_dy_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NULL(scnn_layer_backward(layer, NULL));

    scnn_layer_free(&layer);
}
