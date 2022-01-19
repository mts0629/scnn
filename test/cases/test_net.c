/**
 * @file test_net.c
 * @brief unit test of net.c
 * 
 */
#include "net.h"
#include "fc.h"
#include "sigmoid.h"
#include "softmax.h"

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
