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
    LayerParameter param = { .in = 2, .out = 10 };
    Layer *fc = fc_alloc(param);

    TEST_ASSERT_NOT_NULL(fc);

    TEST_ASSERT_EQUAL_INT(param.in, fc->x_dim[1]);
    TEST_ASSERT_EQUAL_INT(param.in, fc->x_size);

    TEST_ASSERT_EQUAL_INT(param.out, fc->y_dim[1]);
    TEST_ASSERT_EQUAL_INT(param.out, fc->y_size);
    TEST_ASSERT_NOT_NULL(fc->y);

    TEST_ASSERT_NOT_NULL(fc->w);
    TEST_ASSERT_EQUAL_INT((param.in * param.out), fc->w_size);

    TEST_ASSERT_NOT_NULL(fc->b);
    TEST_ASSERT_EQUAL_INT(param.out, fc->b_size);

    TEST_ASSERT_NULL(fc->prev);
    TEST_ASSERT_NULL(fc->next);

    TEST_ASSERT_NOT_NULL(fc->forward);

    layer_free(&fc);

    TEST_ASSERT_NULL(fc);
}

TEST(fc, fc_alloc_invalid_param)
{
    LayerParameter param = { .in = 0, .out = 10 };
    Layer *fc = fc_alloc(param);

    TEST_ASSERT_NULL(fc);

    param.in = 2;
    param.out = 0;
    fc = fc_alloc(param);

    TEST_ASSERT_NULL(fc);
}

TEST(fc, fc_forward)
{
    LayerParameter param = { .in = 2, .out = 3 };
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

    mat_copy(w, 2, 3, fc->w);
    mat_copy(b, 1, 3, fc->b);

    fc->forward(fc, x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(ans, fc->y, (1 * 3));

    layer_free(&fc);
}

TEST(fc, fc_backward)
{
    LayerParameter param = { .in = 2, .out = 3 };
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

    mat_copy(w, 2, 3, fc->w);
    mat_copy(b, 1, 3, fc->b);

    fc->forward(fc, x);

    float dy[] = {
        8, 12, 16
    };

    fc->backward(fc, dy);

    float dx_ans[] = {
        44, 152,
    };

    float dw_ans[] = {
        8, 12, 16,
        8, 12, 16
    };

    float db_ans[] = {
        8, 12, 16
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx_ans, fc->dx, (1 * 2));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dw_ans, fc->dw, (2 * 3));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(db_ans, fc->db, (1 * 3));
}

TEST(fc, fc_update)
{
    LayerParameter param = { .in = 2, .out = 3 };
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

    mat_copy(w, 2, 3, fc->w);
    mat_copy(b, 1, 3, fc->b);

    fc->forward(fc, x);

    float dy[] = {
        8, 12, 16
    };

    fc->backward(fc, dy);

    fc->update(fc, 0.01);

    float w_updated_ans[] = {
        -0.08, 0.88, 1.84,
        2.92, 3.88, 4.84
        //3 4 5
        //0.08, 0.12, 0.16
    };

    float b_updated_ans[] = {
        0.92, 0.88, 0.84
        //1, 1, 1
        //0.08, 0.12, 0.16
    };

    //float w[] = {
    //    0, 1, 2,
    //    3, 4, 5
    //};

    //float b[] = {
    //    1, 1, 1
    //};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(w_updated_ans, fc->w, (2 * 3));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(b_updated_ans, fc->b, (1 * 3));
}
