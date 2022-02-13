/**
 * @file test_sigmoid.c
 * @brief unit tests of sigmoid.c
 * 
 */
#include "sigmoid.h"

#include "mat.h"

#include "unity_fixture.h"

TEST_GROUP(sigmoid);

TEST_SETUP(sigmoid)
{}

TEST_TEAR_DOWN(sigmoid)
{}

TEST(sigmoid, sigmoid_alloc_and_free)
{
    LayerParameter param = { .name = "sigmoid", .in = 10 };
    Layer *sigmoid = sigmoid_alloc(param);

    TEST_ASSERT_NOT_NULL(sigmoid);

    TEST_ASSERT_EQUAL_CHAR_ARRAY(param.name, sigmoid->name, sizeof(param.name));

    TEST_ASSERT_EQUAL_INT(param.in, sigmoid->x_dim[1]);
    TEST_ASSERT_EQUAL_INT(param.in, sigmoid->x_size);
    TEST_ASSERT_NULL(sigmoid->x);

    TEST_ASSERT_EQUAL_INT(param.in, sigmoid->y_dim[1]);
    TEST_ASSERT_EQUAL_INT(param.in, sigmoid->y_size);
    TEST_ASSERT_NOT_NULL(sigmoid->y);

    TEST_ASSERT_NULL(sigmoid->w);
    TEST_ASSERT_EQUAL_INT(0, sigmoid->w_size);
    TEST_ASSERT_NULL(sigmoid->b);
    TEST_ASSERT_EQUAL_INT(0, sigmoid->b_size);

    TEST_ASSERT_NOT_NULL(sigmoid->dx);
    TEST_ASSERT_NULL(sigmoid->dw);
    TEST_ASSERT_NULL(sigmoid->db);

    TEST_ASSERT_NULL(sigmoid->prev);
    TEST_ASSERT_NULL(sigmoid->next);

    TEST_ASSERT_NOT_NULL(sigmoid->forward);
    TEST_ASSERT_NOT_NULL(sigmoid->backward);

    layer_free(&sigmoid);

    TEST_ASSERT_NULL(sigmoid);
}

TEST(sigmoid, sigmoid_alloc_invalid_param)
{
    LayerParameter param = { .name = "sigmoid", .in = 0 };
    Layer *sigmoid = sigmoid_alloc(param);

    TEST_ASSERT_NULL(sigmoid);
}

TEST(sigmoid, sigmoid_forward)
{
    LayerParameter param = { .name = "sigmoid", .in = 11 };
    Layer *sigmoid = sigmoid_alloc(param);

    float x[] = {
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5
    };
    
    float ans[] = {
        0.00669285, 0.0179862, 0.0474259, 0.119203, 0.268941, 0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307
    };

    sigmoid->forward(sigmoid, x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, sigmoid->y, (1 * 11));

    layer_free(&sigmoid);
}

TEST(sigmoid, sigmoid_backward)
{
    LayerParameter param = { .name = "sigmoid", .in = 11 };
    Layer *sigmoid = sigmoid_alloc(param);

    float x[] = {
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5
    };

    sigmoid->forward(sigmoid, x);

    float dy[] = {
        0.0133857, 0.03597242, 0.09485175, 0.23840584, 0.53788284, 1, 1.46211716, 1.76159416, 1.90514825, 1.96402758, 1.9866143
    };

    sigmoid->backward(sigmoid, dy);
    
    float dx_ans[] = {
        8.89889045e-5, 6.35370285e-4, 4.28508507e-3, 2.50310843e-2, 1.05754186e-1, 2.5e-1, 2.87469681e-1, 1.84956086e-1, 8.60682344e-2, 3.46900421e-2, 1.32071244e-2
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx_ans, sigmoid->dx, 11);

    layer_free(&sigmoid);
}
