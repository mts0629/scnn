/**
 * @file test_fc.c
 * @brief unit tests of fc.c
 * 
 */
#include "fc.h"

#include "mat.h"

#include "unity_fixture.h"

TEST_GROUP(fc);

TEST_SETUP(fc)
{}

TEST_TEAR_DOWN(fc)
{}

TEST(fc, fc_alloc_and_free)
{
    LayerParameter param = { .name = "fc", .in = 2, .out = 10 };
    Layer *fc = fc_alloc(param);

    TEST_ASSERT_NOT_NULL(fc);

    TEST_ASSERT_EQUAL_CHAR_ARRAY(param.name, fc->name, sizeof(param.name));

    TEST_ASSERT_EQUAL_INT(param.in, fc->in);
    TEST_ASSERT_NOT_NULL(fc->x);

    TEST_ASSERT_EQUAL_INT(param.out, fc->out);
    TEST_ASSERT_NOT_NULL(fc->y);

    TEST_ASSERT_NOT_NULL(fc->w);

    TEST_ASSERT_NOT_NULL(fc->b);

    TEST_ASSERT_NULL(fc->prev);
    TEST_ASSERT_NULL(fc->next);

    TEST_ASSERT_NOT_NULL(fc->forward);

    layer_free(&fc);

    TEST_ASSERT_NULL(fc);
}

TEST(fc, fc_forward)
{
    LayerParameter param = { .name = "fc", .in = 2, .out = 3 };
    Layer *fc = fc_alloc(param);

    float x[] = {
        1, 1
    };

    float w[] = {
        0, 1, 2,
        3, 4, 5
    };

    float b[] = {
        1, 1, 1
    };

    float ans[] = {
        4, 6, 8
    };

    mat_copy(x, 1, 2, fc->x);
    mat_copy(w, 2, 3, fc->w);
    mat_copy(b, 1, 3, fc->b);

    fc->forward(fc);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, fc->y, (1 * 3));

    layer_free(&fc);
}
