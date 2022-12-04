/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"
#include "scnn_layers.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_net);

scnn_net *net;

TEST_SETUP(scnn_net)
{
    net = NULL;
}

TEST_TEAR_DOWN(scnn_net)
{
    scnn_net_free(&net);

    TEST_ASSERT_NULL(net);
}

TEST(scnn_net, allocate)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL(0, net->size);

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        TEST_ASSERT_NULL(net->layers[i]);
    }

    TEST_ASSERT_NULL(net->input);
    TEST_ASSERT_NULL(net->output);
}

TEST(scnn_net, free_with_NULL)
{
    // free in TEST_TEAR_DOWN
}

TEST(scnn_net, append_layer)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));

    TEST_ASSERT_EQUAL(1, net->size);

    TEST_ASSERT_EQUAL_PTR(fc, net->layers[0]);

    TEST_ASSERT_EQUAL_PTR(fc, net->input);
    TEST_ASSERT_EQUAL_PTR(fc, net->output);
}

TEST(scnn_net, append_2layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, sigmoid));

    TEST_ASSERT_EQUAL(2, net->size);

    TEST_ASSERT_EQUAL_PTR(fc, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->layers[1]);

    TEST_ASSERT_EQUAL_PTR(fc, net->input);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->output);
}

TEST(scnn_net, append_3layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc1 = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });
    scnn_layer *fc2 = scnn_fc_layer((scnn_layer_params){ 0 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc1));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc2));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, sigmoid));

    TEST_ASSERT_EQUAL(3, net->size);

    TEST_ASSERT_EQUAL_PTR(fc1, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(fc2, net->layers[1]);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->layers[2]);

    TEST_ASSERT_EQUAL_PTR(fc1, net->input);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->output);
}

TEST(scnn_net, cannot_append_if_net_is_NULL)
{
    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });

    TEST_ASSERT_NULL(scnn_net_append(NULL, fc));

    scnn_layer_free(&fc);
}

TEST(scnn_net, cannot_append_if_layer_is_NULL)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_append(net, NULL));
}

TEST(scnn_net, cannot_append_if_over_max_size)
{
    net = scnn_net_alloc();

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=2 });
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));
    }

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=2 });
    TEST_ASSERT_NULL(scnn_net_append(net, fc));

    scnn_layer_free(&fc);
}

TEST(scnn_net, init_layer)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });

    scnn_net_append(net, fc);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    TEST_ASSERT_NOT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->x->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[0]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[2]);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->x->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->y->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w->data);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->w->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->w->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->w->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->w->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->b->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dx->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->x->shape, net->layers[0]->dx->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->w->shape, net->layers[0]->dw->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[0]->db);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->b->shape, net->layers[0]->db->shape, 4);
}

TEST(scnn_net, init_2layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    // fc layer
    TEST_ASSERT_NOT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->x->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[0]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[2]);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->x->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->y->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w->data);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->w->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->w->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->w->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->w->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->b->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dx->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->x->shape, net->layers[0]->dx->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->w->shape, net->layers[0]->dw->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[0]->db);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->b->shape, net->layers[0]->db->shape, 4);

    // sigmoid layer
    TEST_ASSERT_NOT_NULL(net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->x->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->y->shape, net->layers[1]->x->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[1]->x->shape, net->layers[1]->y->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[1]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[1]->dx->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[1]->x->shape, net->layers[1]->dx->shape, 4);

    TEST_ASSERT_NULL(net->layers[1]->w);
    TEST_ASSERT_NULL(net->layers[1]->b);
    TEST_ASSERT_NULL(net->layers[1]->dw);
    TEST_ASSERT_NULL(net->layers[1]->db);
}

TEST(scnn_net, init_3layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    // fc layer
    TEST_ASSERT_NOT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->x->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[0]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->x->shape[2]);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->x->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->y->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->y->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w->data);
    TEST_ASSERT_EQUAL_INT(2, net->layers[0]->w->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->w->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->w->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->w->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b->data);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, net->layers[0]->b->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, net->layers[0]->b->shape[3]);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dx->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->x->shape, net->layers[0]->dx->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->w->shape, net->layers[0]->dw->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[0]->db);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->b->shape, net->layers[0]->db->shape, 4);

    // sigmoid layer
    TEST_ASSERT_NOT_NULL(net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->x->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[0]->y->shape, net->layers[1]->x->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[1]->x->shape, net->layers[1]->y->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[1]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[1]->dx->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[1]->x->shape, net->layers[1]->dx->shape, 4);

    TEST_ASSERT_NULL(net->layers[1]->w);
    TEST_ASSERT_NULL(net->layers[1]->b);
    TEST_ASSERT_NULL(net->layers[1]->dw);
    TEST_ASSERT_NULL(net->layers[1]->db);

    // softmax layer
    TEST_ASSERT_NOT_NULL(net->layers[2]->x);
    TEST_ASSERT_NOT_NULL(net->layers[2]->x->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[1]->y->shape, net->layers[2]->x->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[2]->y);
    TEST_ASSERT_NOT_NULL(net->layers[2]->y->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[2]->x->shape, net->layers[2]->y->shape, 4);

    TEST_ASSERT_NOT_NULL(net->layers[2]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[2]->dx->data);
    TEST_ASSERT_EQUAL_INT_ARRAY(net->layers[2]->x->shape, net->layers[2]->dx->shape, 4);

    TEST_ASSERT_NULL(net->layers[2]->w);
    TEST_ASSERT_NULL(net->layers[2]->b);
    TEST_ASSERT_NULL(net->layers[2]->dw);
    TEST_ASSERT_NULL(net->layers[2]->db);
}

/*TEST(scnn_net, forward)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(fc->w, (float[]){ 1, 2, 3, 4 }, fc->w->size);
    scnn_mat_copy_from_array(fc->b, (float[]){ 0, 1 }, fc->w->size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_forward(net, x);

    float y_fc[] = { 0.7, 2 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_fc, net->layers[0]->y->data, net->layers[0]->y->size);

    float y_sigmoid[] = { 0.66818777, 0.88079708 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_sigmoid, net->layers[1]->y->data, net->layers[1]->y->size);

    float y_softmax[] = { 0.44704699, 0.55295301 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_softmax, net->layers[2]->y->data, net->layers[2]->y->size);

    scnn_net_free(&net);
}

TEST(scnn_net, forward_net_is_null)
{
    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_forward(NULL, x);
}

TEST(scnn_net, forward_x_is_null)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(fc->w, (float[]){ 1, 2, 3, 4 }, fc->w->size);
    scnn_mat_copy_from_array(fc->b, (float[]){ 0, 1 }, fc->w->size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat_fill(net->output->y, 0);

    scnn_net_forward(net, NULL);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, net->output->y->data, net->output->y->size);

    scnn_net_free(&net);
}

TEST(scnn_net, backward)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(fc->w, (float[]){ 1, 2, 3, 4 }, fc->w->size);
    scnn_mat_copy_from_array(fc->b, (float[]){ 0, 1 }, fc->w->size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_forward(net, x);

    scnn_mat *t = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(t, (float[]){ 0, 1 }, x->size);

    scnn_net_backward(net, t);

    float dx_ans[] = {
        0.00524194, 0.10959995
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx_ans, net->layers[0]->dx->data, net->layers[0]->y->size);

    scnn_net_free(&net);
}

TEST(scnn_net, backward_net_is_null)
{
    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 }});
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_backward(NULL, x);
}

TEST(scnn_net, backward_t_is_null)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(fc->w, (float[]){ 1, 2, 3, 4 }, fc->w->size);
    scnn_mat_copy_from_array(fc->b, (float[]){ 0, 1 }, fc->w->size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_mat_fill(net->input->dx, 0);

    scnn_net_forward(net, x);

    scnn_net_backward(net, NULL);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, net->input->dx->data, net->input->dx->size);

    scnn_net_free(&net);
}*/
