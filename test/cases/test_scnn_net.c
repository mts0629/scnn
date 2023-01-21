/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"
#include "scnn_layers.h"
#include "scnn_blas.h"

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

    TEST_ASSERT_EQUAL(0, scnn_net_size(net));

    TEST_ASSERT_EQUAL(1, net->batch_size);

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        TEST_ASSERT_NULL(net->layers[i]);
    }

    TEST_ASSERT_NULL(scnn_net_input(net));
    TEST_ASSERT_NULL(scnn_net_output(net));
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

    TEST_ASSERT_EQUAL(1, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(fc, net->layers[0]);

    TEST_ASSERT_EQUAL_PTR(fc, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(fc, scnn_net_output(net));
}

TEST(scnn_net, append_2layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, sigmoid));

    TEST_ASSERT_EQUAL(2, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(fc, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->layers[1]);

    TEST_ASSERT_EQUAL_PTR(fc, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(sigmoid, scnn_net_output(net));
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

    TEST_ASSERT_EQUAL(3, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(fc1, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(fc2, net->layers[1]);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->layers[2]);

    TEST_ASSERT_EQUAL_PTR(fc1, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(sigmoid, scnn_net_output(net));
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

/**
 * @brief Check matrix shape
 * 
 * @param expect Expected shape
 * @param actual Actual shape
 */
static void check_mat_shape(const int *expect, const int *actual)
{
    TEST_ASSERT_EQUAL_INT(expect[0], actual[0]);
    TEST_ASSERT_EQUAL_INT(expect[1], actual[1]);
    TEST_ASSERT_EQUAL_INT(expect[2], actual[2]);
    TEST_ASSERT_EQUAL_INT(expect[3], actual[3]);
}

TEST(scnn_net, init_layer)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });

    scnn_net_append(net, fc);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    TEST_ASSERT_NOT_NULL(net->layers[0]->x);
    TEST_ASSERT_NOT_NULL(net->layers[0]->x->data);
    check_mat_shape(scnn_shape(1, 1, 1, 2), net->layers[0]->x->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y->data);
    check_mat_shape(scnn_shape(1, 10, 1, 1), net->layers[0]->y->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w->data);
    check_mat_shape(scnn_shape(2, 10, 1, 1), net->layers[0]->w->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b->data);
    check_mat_shape(scnn_shape(1, 10, 1, 1), net->layers[0]->b->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dx->data);
    check_mat_shape(net->layers[0]->x->shape, net->layers[0]->dx->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw->data);
    check_mat_shape(net->layers[0]->w->shape, net->layers[0]->dw->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->db);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db->data);
    check_mat_shape(net->layers[0]->b->shape, net->layers[0]->db->shape);
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
    check_mat_shape(scnn_shape(1, 1, 1, 2), net->layers[0]->x->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y->data);
    check_mat_shape(scnn_shape(1, 10, 1, 1), net->layers[0]->y->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w->data);
    check_mat_shape(scnn_shape(2, 10, 1, 1), net->layers[0]->w->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b->data);
    check_mat_shape(scnn_shape(1, 10, 1, 1), net->layers[0]->b->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dx->data);
    check_mat_shape(net->layers[0]->x->shape, net->layers[0]->dx->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw->data);
    check_mat_shape(net->layers[0]->w->shape, net->layers[0]->dw->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->db);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db->data);
    check_mat_shape(net->layers[0]->b->shape, net->layers[0]->db->shape);

    // sigmoid layer
    TEST_ASSERT_NOT_NULL(net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->x->data);
    check_mat_shape(net->layers[0]->y->shape, net->layers[1]->x->shape);

    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y->data);
    check_mat_shape(net->layers[1]->x->shape, net->layers[1]->y->shape);

    TEST_ASSERT_NOT_NULL(net->layers[1]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[1]->dx->data);
    check_mat_shape(net->layers[1]->x->shape, net->layers[1]->dx->shape);

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
    check_mat_shape(scnn_shape(1, 1, 1, 2), net->layers[0]->x->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->y);
    TEST_ASSERT_NOT_NULL(net->layers[0]->y->data);
    check_mat_shape(scnn_shape(1, 10, 1, 1), net->layers[0]->y->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->w);
    TEST_ASSERT_NOT_NULL(net->layers[0]->w->data);
    check_mat_shape(scnn_shape(2, 10, 1, 1), net->layers[0]->w->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->b);
    TEST_ASSERT_NOT_NULL(net->layers[0]->b->data);
    check_mat_shape(scnn_shape(1, 10, 1, 1), net->layers[0]->b->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dx->data);
    check_mat_shape(net->layers[0]->x->shape, net->layers[0]->dx->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->dw);
    TEST_ASSERT_NOT_NULL(net->layers[0]->dw->data);
    check_mat_shape(net->layers[0]->w->shape, net->layers[0]->dw->shape);

    TEST_ASSERT_NOT_NULL(net->layers[0]->db);
    TEST_ASSERT_NOT_NULL(net->layers[0]->db->data);
    check_mat_shape(net->layers[0]->b->shape, net->layers[0]->db->shape);

    // sigmoid layer
    TEST_ASSERT_NOT_NULL(net->layers[1]->x);
    TEST_ASSERT_NOT_NULL(net->layers[1]->x->data);
    check_mat_shape(net->layers[0]->y->shape, net->layers[1]->x->shape);

    TEST_ASSERT_NOT_NULL(net->layers[1]->y);
    TEST_ASSERT_NOT_NULL(net->layers[1]->y->data);
    check_mat_shape(net->layers[1]->x->shape, net->layers[1]->y->shape);

    TEST_ASSERT_NOT_NULL(net->layers[1]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[1]->dx->data);
    check_mat_shape(net->layers[1]->x->shape, net->layers[1]->dx->shape);

    TEST_ASSERT_NULL(net->layers[1]->w);
    TEST_ASSERT_NULL(net->layers[1]->b);
    TEST_ASSERT_NULL(net->layers[1]->dw);
    TEST_ASSERT_NULL(net->layers[1]->db);

    // softmax layer
    TEST_ASSERT_NOT_NULL(net->layers[2]->x);
    TEST_ASSERT_NOT_NULL(net->layers[2]->x->data);
    check_mat_shape(net->layers[1]->y->shape, net->layers[2]->x->shape);

    TEST_ASSERT_NOT_NULL(net->layers[2]->y);
    TEST_ASSERT_NOT_NULL(net->layers[2]->y->data);
    check_mat_shape(net->layers[2]->x->shape, net->layers[2]->y->shape);

    TEST_ASSERT_NOT_NULL(net->layers[2]->dx);
    TEST_ASSERT_NOT_NULL(net->layers[2]->dx->data);
    check_mat_shape(net->layers[2]->x->shape, net->layers[2]->dx->shape);

    TEST_ASSERT_NULL(net->layers[2]->w);
    TEST_ASSERT_NULL(net->layers[2]->b);
    TEST_ASSERT_NULL(net->layers[2]->dw);
    TEST_ASSERT_NULL(net->layers[2]->db);
}

TEST(scnn_net, cannot_init_if_net_is_NULL)
{
    TEST_ASSERT_NULL(scnn_net_init(NULL));
}

TEST(scnn_net, cannot_init_if_size_is_0)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_init(net));
}

TEST(scnn_net, cannot_init_if_in_shape_is_invalid)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .out=10 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    TEST_ASSERT_NULL(scnn_net_init(net));
}

TEST(scnn_net, cannot_init_if_contains_invalid_layer)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 2 }, .out=10 });
    scnn_layer *fc2 = scnn_fc_layer((scnn_layer_params){ 0 });
    scnn_layer *sigmoid = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, fc2);
    scnn_net_append(net, sigmoid);

    TEST_ASSERT_NULL(scnn_net_init(net));
}

/**
 * @brief Copy contents of array to scnn_mat
 * 
 * @param src Source array
 * @param dst Destination matrix
 */
static void copy_data_to_mat(const scnn_dtype *src, const scnn_mat *dst)
{
    scnn_scopy(dst->size, src, 1, dst->data, 1);
}

/**
 * @brief Declearation and initialize data array
 * 
 */
#define MAT_DATA(...) (scnn_dtype[]){ __VA_ARGS__ }

/**
 * @brief Check network output with an expectation
 * 
 * @param expect    Expected result
 * @param net       Network which output will be checked
 */
static void check_net_output(const scnn_dtype *expect, const scnn_net *net)
{
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expect, scnn_net_output(net)->y->data, net->output->y->size);
}

TEST(scnn_net, forward_layer)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });

    scnn_net_append(net, fc);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    scnn_net_forward(net, MAT_DATA(0.1, 0.2));

    check_net_output(MAT_DATA(0.7, 2), net);
}

TEST(scnn_net, forward_2layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    scnn_net_forward(net, MAT_DATA(0.1, 0.2));

    check_net_output(MAT_DATA(0.668188, 0.880797), net);
}

TEST(scnn_net, forward_3layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    scnn_net_forward(net, MAT_DATA(0.1, 0.2));

    check_net_output(MAT_DATA(0.44704707, 0.55295293), net);
}

TEST(scnn_net, forward_failed_when_net_is_NULL)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });

    scnn_net_append(net, fc);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    copy_data_to_mat(MAT_DATA(-1, -1), scnn_net_output(net)->y);

    scnn_net_forward(NULL, MAT_DATA(0.1, 0.2));

    check_net_output(MAT_DATA(-1, -1), net);
}

TEST(scnn_net, forward_failed_when_x_is_NULL)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });

    scnn_net_append(net, fc);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    copy_data_to_mat(MAT_DATA(-1, -1), scnn_net_output(net)->y);

    scnn_net_forward(net, NULL);

    check_net_output(MAT_DATA(-1, -1), net);
}

TEST(scnn_net, backward_3layers)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    scnn_net_forward(net, MAT_DATA(0.1, 0.2));

    const int size = scnn_net_output(net)->y->size;

    scnn_dtype dy[2];
    scnn_scopy(size, scnn_net_output(net)->y->data, 1, dy, 1);
    scnn_saxpy(size, -1, MAT_DATA(0, 1), 1, dy, 1);

    scnn_net_backward(net, dy);

    // fc
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(MAT_DATA(0.00524194, 0.10959995), net->layers[0]->dx->data, net->layers[0]->dx->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        MAT_DATA(0.00991160, -0.00469371, 0.01982320, -0.00938742), net->layers[0]->dw->data, net->layers[0]->dw->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(MAT_DATA(0.09911601, -0.04693710), net->layers[0]->db->data, net->layers[0]->db->size);

    // sigmoid
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(MAT_DATA(0.09911601, -0.04693710), net->layers[1]->dx->data, net->layers[1]->dx->size);

    // softmax
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(MAT_DATA(0.44704707, -0.44704699), net->layers[2]->dx->data, net->layers[2]->dx->size);
}

TEST(scnn_net, backward_failed_when_net_is_NULL)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    scnn_mat_fill(net->layers[0]->dx, -1);
    scnn_mat_fill(net->layers[0]->dw, -1);
    scnn_mat_fill(net->layers[0]->db, -1);

    scnn_mat_fill(net->layers[1]->dx, -1);

    scnn_mat_fill(net->layers[2]->dx, -1);

    scnn_net_forward(net, MAT_DATA(0.1, 0.2));

    const int size = scnn_net_output(net)->y->size;

    scnn_dtype dy[2];
    scnn_scopy(size, scnn_net_output(net)->y->data, 1, dy, 1);
    scnn_saxpy(size, -1, MAT_DATA(0, 1), 1, dy, 1);

    scnn_net_backward(NULL, dy);

    // fc
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[0]->dx->data, net->layers[0]->dx->size);
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[0]->dw->data, net->layers[0]->dw->size);
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[0]->db->data, net->layers[0]->db->size);
    // sigmoid
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[1]->dx->data, net->layers[1]->dx->size);
    // softmax
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[2]->dx->data, net->layers[2]->dx->size);
}

TEST(scnn_net, backward_failed_when_dy_is_NULL)
{
    net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in_shape={ 1, 2, 1, 1 }, .out=2 });
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ 0 });
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ 0 });

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_net_init(net);

    copy_data_to_mat(MAT_DATA(1, 2, 3, 4), fc->w);
    copy_data_to_mat(MAT_DATA(0, 1), fc->b);

    scnn_mat_fill(net->layers[0]->dx, -1);
    scnn_mat_fill(net->layers[0]->dw, -1);
    scnn_mat_fill(net->layers[0]->db, -1);

    scnn_mat_fill(net->layers[1]->dx, -1);

    scnn_mat_fill(net->layers[2]->dx, -1);

    scnn_net_forward(net, MAT_DATA(0.1, 0.2));

    scnn_net_backward(net, NULL);

    // fc
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[0]->dx->data, net->layers[0]->dx->size);
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[0]->dw->data, net->layers[0]->dw->size);
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[0]->db->data, net->layers[0]->db->size);
    // sigmoid
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[1]->dx->data, net->layers[1]->dx->size);
    // softmax
    TEST_ASSERT_EACH_EQUAL_FLOAT(-1, net->layers[2]->dx->data, net->layers[2]->dx->size);
}
