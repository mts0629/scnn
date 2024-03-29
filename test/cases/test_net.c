/**
 * @file test_net.c
 * @brief unit test of net.c
 * 
 */
#include "data.h"
#include "net.h"
#include "layers.h"
#include "mat.h"
#include "random.h"
#include "util.h"

#include "unity_fixture.h"

TEST_GROUP(net);

TEST_SETUP(net)
{}

TEST_TEAR_DOWN(net)
{}

TEST(net, net_alloc_and_free)
{
    Net *net = net_alloc();

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL(0, net->size);

    for (int i = 0; i < NET_LAYER_MAX_SIZE; i++) {
        TEST_ASSERT_NULL(net->layers[i]);
    }

    TEST_ASSERT_NULL(net->input_layer);
    TEST_ASSERT_NULL(net->output_layer);

    net_free(&net);

    TEST_ASSERT_NULL(net);
}

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

    TEST_ASSERT_EQUAL_INT(3, net->size);

    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->x_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->y_size);
    TEST_ASSERT_EQUAL_INT((2 * 10), net->layers[0]->w_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->b_size);
    TEST_ASSERT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_EQUAL_INT(-1, net->layers[0]->prev_id);
    TEST_ASSERT_EQUAL_INT(net->layers[1]->id, net->layers[0]->next_id);

    TEST_ASSERT_EQUAL_INT(10, net->layers[1]->x_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[1]->y_size);
    TEST_ASSERT_EQUAL_PTR(net->layers[0]->y, net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_EQUAL_INT(net->layers[0]->id, net->layers[1]->prev_id);
    TEST_ASSERT_EQUAL_INT(net->layers[2]->id, net->layers[1]->next_id);

    TEST_ASSERT_EQUAL_INT(10, net->layers[2]->x_size);
    TEST_ASSERT_EQUAL_INT(10, net->layers[2]->y_size);
    TEST_ASSERT_EQUAL_PTR(net->layers[1]->y, net->layers[2]->x);
    TEST_ASSERT_NOT_NULL(net->layers[2]->y);
    TEST_ASSERT_EQUAL_INT(net->layers[1]->id, net->layers[2]->prev_id);
    TEST_ASSERT_EQUAL_INT(-1, net->layers[2]->next_id);

    TEST_ASSERT_EQUAL_PTR(net->layers[0], net->input_layer);
    TEST_ASSERT_EQUAL_PTR(net->layers[2], net->output_layer);

    net_free(&net);

    TEST_ASSERT_NULL(net);
}

TEST(net, net_create_over_size)
{
    Net *net = net_create(
        257,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=2, .out=10 }),
            sigmoid_layer((LayerParameter){ .in=10 }),
            softmax_layer((LayerParameter){ .in=10 })
        }
    );

    TEST_ASSERT_NULL(net);
}

TEST(net, net_append)
{
    Net *net = net_alloc();

    Layer *fc1 = fc_layer((LayerParameter){ .in=2, .out=10 });
    Layer *fc2 = fc_layer((LayerParameter){ .in=10, .out=2 });

    TEST_ASSERT_EQUAL(net, net_append(net, fc1));
    TEST_ASSERT_EQUAL(net, net_append(net, fc2));

    TEST_ASSERT_EQUAL(2, net->size);

    TEST_ASSERT_EQUAL_PTR(fc1, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(fc2, net->layers[1]);

    TEST_ASSERT_EQUAL_PTR(fc1, net->input_layer);
    TEST_ASSERT_EQUAL_PTR(fc2, net->output_layer);

    net_free(&net);
}

TEST(net, net_append_null)
{
    Net *net = net_alloc();

    TEST_ASSERT_NULL(net_append(net, NULL));
}

TEST(net, net_init_layer_params)
{
#define IN_SIZE 2
#define MID_SIZE 3

    Net *net = net_create(
        3,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=IN_SIZE, .out=MID_SIZE }),
            sigmoid_layer((LayerParameter){ .in=MID_SIZE }),
            softmax_layer((LayerParameter){ .in=MID_SIZE })
        }
    );

    Layer *fc = net->layers[0];

    rand_seed(0);

    float rand_vals[IN_SIZE * MID_SIZE];
    float scale = 1.0f / sqrt(1.0f / fc->x_size);
    for (int i = 0; i < fc->w_size; i++) {
        rand_vals[i] = rand_norm(0, 1) * scale;
    }

    rand_seed(0);

    net_init_layer_params(net);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rand_vals, fc->w, fc->w_size);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, fc->b, fc->b_size);

    net_free(&net);

#undef IN_SIZE
#undef MID_SIZE
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
    fdata_copy(w, net->layers[0]->w_size, net->layers[0]->w);

    float b[] = { 0, 1 };
    fdata_copy(b, net->layers[0]->b_size, net->layers[0]->b);

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
    fdata_copy(w, net->layers[0]->w_size, net->layers[0]->w);

    float b[] = { 0, 1 };
    fdata_copy(b, net->layers[0]->b_size, net->layers[0]->b);

    float x[] = { 0.1, 0.2 };

    net_forward(net, x);

    float t[] = { 0, 1 };

    net_backward(net, t);

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
