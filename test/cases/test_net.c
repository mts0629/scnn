/**
 * @file test_net.c
 * @brief unit test of net.c
 * 
 */
#include "net.h"
#include "layers.h"
#include "mat.h"

#include "unity_fixture.h"

TEST_GROUP(net);

TEST_SETUP(net)
{}

TEST_TEAR_DOWN(net)
{}

TEST(net, net_create_and_free)
{
    Net *net = net_create(
        3,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=2, .out=10 }),
            sigmoid_layer((LayerParameter){ .in=10 }),
            softmax_layer((LayerParameter){ .in=10 })
        }
    );

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_NOT_NULL(net->layers);

    TEST_ASSERT_EQUAL_INT(3, net->length);

    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->x_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->y_size);
    TEST_ASSERT_EQUAL_INT((2 * 10), net->layers[0]->w_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->b_size);
    TEST_ASSERT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NULL(net->layers[0]->prev);
    TEST_ASSERT_EQUAL_PTR(net->layers[1], net->layers[0]->next);

    TEST_ASSERT_EQUAL_INT(10, net->layers[1]->x_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[1]->y_size);
    TEST_ASSERT_EQUAL_PTR(net->layers[0]->y, net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_EQUAL_PTR(net->layers[0], net->layers[1]->prev);
    TEST_ASSERT_EQUAL_PTR(net->layers[2], net->layers[1]->next);

    TEST_ASSERT_EQUAL_INT(10, net->layers[2]->x_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[2]->y_size);
    TEST_ASSERT_EQUAL_PTR(net->layers[1]->y, net->layers[2]->x);
    TEST_ASSERT_NOT_NULL(net->layers[2]->y);
    TEST_ASSERT_EQUAL_PTR(net->layers[1], net->layers[2]->prev);
    TEST_ASSERT_NULL(net->layers[2]->next);

    TEST_ASSERT_EQUAL_PTR(net->layers[0], net->input_layer);
    TEST_ASSERT_EQUAL_PTR(net->layers[2], net->output_layer);

    net_free(&net);

    TEST_ASSERT_NULL(net);
}

TEST(net, net_forward)
{
    Net *net = net_create(
        3,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=2, .out=2 }),
            sigmoid_layer((LayerParameter){ .in=2 }),
            softmax_layer((LayerParameter){ .in=2 })
        }
    );

    float w[] = {
        1, 2,
        3, 4
    };
    mat_copy(w, 2, 2, net->layers[0]->w);

    float b[] = { 0, 1 };
    mat_copy(b, 1, 2, net->layers[0]->b);

    float x[] = { 0.1, 0.2 };

    net_forward(net, x);

    float y_fc[] = { 0.7, 2 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_fc, net->layers[0]->y, 2);

    float y_sigmoid[] = { 0.66818777, 0.88079708 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_sigmoid, net->layers[1]->y, 2);

    float y_softmax[] = { 0.44704699, 0.55295301 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_softmax, net->layers[2]->y, 2);

    net_free(&net);
}

TEST(net, net_backward)
{
    Net *net = net_create(
        3,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=2, .out=2 }),
            sigmoid_layer((LayerParameter){ .in=2 }),
            softmax_layer((LayerParameter){ .in=2 })
        }
    );

    float w[] = {
        1, 2,
        3, 4
    };
    mat_copy(w, 2, 2, net->layers[0]->w);

    float b[] = { 0, 1 };
    mat_copy(b, 1, 2, net->layers[0]->b);

    float x[] = { 0.1, 0.2 };

    net_forward(net, x);

    float t[] = { 0, 1 };

    // get diff of network output
    float dy[2];
    mat_sub(net->output_layer->y, t, dy, 1, 2);

    net_backward(net, dy);

    TEST_ASSERT_NOT_NULL(net->layers[2]->dx);

    TEST_ASSERT_NOT_NULL(net->layers[1]->dx);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db);

    float dx_ans[] = {
        0.00524194, 0.10959995
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx_ans, net->layers[0]->dx, (1 * 2));

    net_free(&net);
}
