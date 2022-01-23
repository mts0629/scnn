/**
 * @file test_net.c
 * @brief unit test of net.c
 * 
 */
#include "net.h"
#include "fc.h"
#include "sigmoid.h"
#include "softmax.h"
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
        "net",
        3,
        (Layer*[]){
            fc_alloc((LayerParameter){ .name="fc", .in=2, .out=10 }),
            sigmoid_alloc((LayerParameter){ .name="sigmoid", .in=10 }),
            softmax_alloc((LayerParameter){ .name="softmax", .in=10 })
        }
    );

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL_CHAR_ARRAY("net", net->name, 3);
    TEST_ASSERT_NOT_NULL(net->layers);

    TEST_ASSERT_EQUAL_INT(3, net->num_layers);

    TEST_ASSERT_EQUAL_CHAR_ARRAY("fc", net->layers[0]->name, 2);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->in);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->out);
    TEST_ASSERT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NULL(net->layers[0]->prev);
    TEST_ASSERT_EQUAL_PTR(net->layers[1], net->layers[0]->next);

    TEST_ASSERT_EQUAL_CHAR_ARRAY("sigmoid", net->layers[1]->name, 7);
    TEST_ASSERT_EQUAL_INT(10, net->layers[1]->in);
    TEST_ASSERT_EQUAL_INT(10, net->layers[1]->out);
    TEST_ASSERT_EQUAL_PTR(net->layers[0]->y, net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_EQUAL_PTR(net->layers[0], net->layers[1]->prev);
    TEST_ASSERT_EQUAL_PTR(net->layers[2], net->layers[1]->next);

    TEST_ASSERT_EQUAL_CHAR_ARRAY("softmax", net->layers[2]->name, 7);
    TEST_ASSERT_EQUAL_INT(10, net->layers[2]->in);
    TEST_ASSERT_EQUAL_INT(10, net->layers[2]->out);
    TEST_ASSERT_EQUAL_PTR(net->layers[1]->y, net->layers[2]->x);
    TEST_ASSERT_NOT_NULL(net->layers[2]->y);
    TEST_ASSERT_EQUAL_PTR(net->layers[1], net->layers[2]->prev);
    TEST_ASSERT_NULL(net->layers[2]->next);

    net_free(&net);

    TEST_ASSERT_NULL(net);
}

TEST(net, net_forward)
{
    Net *net = net_create(
        "net",
        3,
        (Layer*[]){
            fc_alloc((LayerParameter){ .name="fc", .in=2, .out=2 }),
            sigmoid_alloc((LayerParameter){ .name="sigmoid", .in=2 }),
            softmax_alloc((LayerParameter){ .name="softmax", .in=2 })
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
        "net",
        3,
        (Layer*[]){
            fc_alloc((LayerParameter){ .name="fc", .in=2, .out=2 }),
            sigmoid_alloc((LayerParameter){ .name="sigmoid", .in=2 }),
            softmax_alloc((LayerParameter){ .name="softmax", .in=2 })
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
    float dy[2];
    for (int i = 0; i < 2; i++)
    {
        dy[i] = net->layers[2]->y[i] - t[i];
    }

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
