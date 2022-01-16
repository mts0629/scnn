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

    TEST_ASSERT_EQUAL_INT(param.in, sigmoid->in);
    TEST_ASSERT_NOT_NULL(sigmoid->x);

    TEST_ASSERT_EQUAL_INT(param.in, sigmoid->out);
    TEST_ASSERT_NOT_NULL(sigmoid->y);

    TEST_ASSERT_NULL(sigmoid->w);

    TEST_ASSERT_NULL(sigmoid->b);

    TEST_ASSERT_NULL(sigmoid->prev);
    TEST_ASSERT_NULL(sigmoid->next);

    TEST_ASSERT_NOT_NULL(sigmoid->forward);

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

    mat_copy(x, 1, 11, sigmoid->x);

    sigmoid->forward(sigmoid);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, sigmoid->y, (1 * 11));

    layer_free(&sigmoid);
}
