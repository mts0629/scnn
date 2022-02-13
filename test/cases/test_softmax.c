/**
 * @file test_softmax.c
 * @brief unit tests of softmax.c
 * 
 */
#include "softmax.h"

#include "mat.h"

#include "unity_fixture.h"

TEST_GROUP(softmax);

TEST_SETUP(softmax)
{}

TEST_TEAR_DOWN(softmax)
{}

TEST(softmax, softmax_alloc_and_free)
{
    LayerParameter param = { .name = "softmax", .in = 10 };
    Layer *softmax = softmax_alloc(param);

    TEST_ASSERT_NOT_NULL(softmax);

    TEST_ASSERT_EQUAL_CHAR_ARRAY(param.name, softmax->name, sizeof(param.name));

    TEST_ASSERT_EQUAL_INT(param.in, softmax->x_dim[1]);
    TEST_ASSERT_EQUAL_INT(param.in, softmax->x_size);
    TEST_ASSERT_NULL(softmax->x);

    TEST_ASSERT_EQUAL_INT(param.in, softmax->y_dim[1]);
    TEST_ASSERT_EQUAL_INT(param.in, softmax->y_size);
    TEST_ASSERT_NOT_NULL(softmax->y);

    TEST_ASSERT_NULL(softmax->w);
    TEST_ASSERT_EQUAL_INT(0, softmax->w_size);
    TEST_ASSERT_NULL(softmax->b);
    TEST_ASSERT_EQUAL_INT(0, softmax->b_size);

    TEST_ASSERT_NOT_NULL(softmax->dx);
    TEST_ASSERT_NULL(softmax->dw);
    TEST_ASSERT_NULL(softmax->db);

    TEST_ASSERT_NULL(softmax->prev);
    TEST_ASSERT_NULL(softmax->next);

    TEST_ASSERT_NOT_NULL(softmax->forward);
    TEST_ASSERT_NOT_NULL(softmax->backward);

    layer_free(&softmax);

    TEST_ASSERT_NULL(softmax);
}

TEST(softmax, softmax_alloc_invalid_param)
{
    LayerParameter param = { .name = "softmax", .in = 0 };
    Layer *softmax = softmax_alloc(param);

    TEST_ASSERT_NULL(softmax);
}


TEST(softmax, softmax_forward)
{
    LayerParameter param = { .name = "softmax", .in = 4 };
    Layer *softmax = softmax_alloc(param);

    float x[] = {
        -1, 0, 3, 5
    };

    float ans[] = {
        0.0021657, 0.00588697, 0.11824302, 0.87370431
    };

    softmax->forward(softmax, x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, softmax->y, (1 * 4));

    layer_free(&softmax);
}

TEST(softmax, softmax_backward)
{
    LayerParameter param = { .name = "softmax", .in = 4 };
    Layer *softmax = softmax_alloc(param);

    float x[] = {
        -1, 0, 3, 5
    };

    softmax->forward(softmax, x);

    float t[] = {
        0, 0, 1, 0
    };

    float dy[4];
    for (int i = 0; i < 4; i++)
    {
        dy[i] = softmax->y[i] - t[i];
    }

    softmax->backward(softmax, dy);

    float dx_ans[] = {
        0.0021657, 0.00588697, -0.88175698, 0.87370431
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx_ans, softmax->dx, (1 * 4));

    layer_free(&softmax);
}
