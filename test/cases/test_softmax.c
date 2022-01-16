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

    TEST_ASSERT_EQUAL_INT(param.in, softmax->in);
    TEST_ASSERT_NOT_NULL(softmax->x);

    TEST_ASSERT_EQUAL_INT(param.in, softmax->out);
    TEST_ASSERT_NOT_NULL(softmax->y);

    TEST_ASSERT_NULL(softmax->w);

    TEST_ASSERT_NULL(softmax->b);

    TEST_ASSERT_NULL(softmax->prev);
    TEST_ASSERT_NULL(softmax->next);

    TEST_ASSERT_NOT_NULL(softmax->forward);

    layer_free(&softmax);

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

    mat_copy(x, 1, 4, softmax->x);

    softmax->forward(softmax);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, softmax->y, (1 * 4));

    layer_free(&softmax);
}
