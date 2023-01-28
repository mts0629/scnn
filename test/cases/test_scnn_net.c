/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"

#include "unity.h"

#include "mock_scnn_layer.h"

scnn_net *net;

void setUp(void)
{
    net = NULL;
}

void tearDown(void)
{
}

void test_allocate_and_free(void)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL_INT(0, scnn_net_size(net));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_batch_size(net));

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        TEST_ASSERT_NULL(scnn_net_layers(net)[i]);
    }

    TEST_ASSERT_NULL(scnn_net_input(net));
    TEST_ASSERT_NULL(scnn_net_output(net));

    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_free_pointer_to_NULL(void)
{
    scnn_net_free(&net);
}

void test_free_NULL(void)
{
    scnn_net_free(NULL);
}

void test_append_layer(void)
{
    net = scnn_net_alloc();

    scnn_layer layer;
    scnn_layer_connect_Expect(NULL, &layer);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &layer));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(&layer, scnn_net_layers(net)[0]);

    TEST_ASSERT_EQUAL_PTR(&layer, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&layer, scnn_net_output(net));

    scnn_layer_free_Expect(&(scnn_net_layers(net)[0]));
    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[3];
    scnn_layer_connect_Expect(NULL, &layers[0]);
    scnn_layer_connect_Expect(&layers[0], &layers[1]);
    scnn_layer_connect_Expect(&layers[1], &layers[2]);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &layers[i]));
    }

    TEST_ASSERT_EQUAL_INT(3, scnn_net_size(net));

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_PTR(&layers[i], scnn_net_layers(net)[i]);
    }

    TEST_ASSERT_EQUAL_PTR(&layers[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&layers[2], scnn_net_output(net));

    for (int i = 0; i < 3; i++) {
        scnn_layer_free_Expect(&(scnn_net_layers(net)[i]));
    }
    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_fail_if_net_is_NULL(void)
{
    scnn_layer layer;
    TEST_ASSERT_NULL(scnn_net_append(NULL, &layer));
}

void test_append_fail_if_layer_is_NULL(void)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_append(net, NULL));

    TEST_ASSERT_EQUAL_INT(0, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(NULL, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(NULL, scnn_net_output(net));

    scnn_net_free(&net);
}

void test_append_fail_if_exeeeds_max_size(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[SCNN_NET_MAX_SIZE];
    scnn_layer_connect_Expect(NULL, &layers[0]);
    for (int i = 1; i < SCNN_NET_MAX_SIZE; i++) {
        scnn_layer_connect_Expect(&layers[i - 1], &layers[i]);
    }
    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &layers[i]));
    }

    TEST_ASSERT_EQUAL_INT(SCNN_NET_MAX_SIZE, scnn_net_size(net));
    TEST_ASSERT_EQUAL_PTR(&layers[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&layers[SCNN_NET_MAX_SIZE - 1], scnn_net_output(net));

    scnn_layer extra_layer;
    TEST_ASSERT_NULL(scnn_net_append(net, &extra_layer));

    TEST_ASSERT_EQUAL_INT(SCNN_NET_MAX_SIZE, scnn_net_size(net));
    TEST_ASSERT_EQUAL_PTR(&layers[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&layers[SCNN_NET_MAX_SIZE - 1], scnn_net_output(net));

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        scnn_layer_free_Expect(&(scnn_net_layers(net)[i]));
    }
    scnn_net_free(&net);

    TEST_ASSERT_NULL(net);
}

void test_init(void)
{
    net = scnn_net_alloc();

    scnn_layer layer;
    scnn_layer_connect_Ignore();
    scnn_net_append(net, &layer);

    scnn_layer_init_ExpectAndReturn(&layer, &layer);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_init_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[3];
    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &layers[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_ExpectAndReturn(&layers[i], &layers[i]);
    }
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_init_fail_if_net_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_net_init(NULL));
}

void test_init_fail_if_net_size_is_0(void)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_init(net));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_init_fail_if_layer_init_fail(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[3];
    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &layers[i]);
    }

    scnn_layer_init_ExpectAndReturn(&layers[0], &layers[0]);
    scnn_layer_init_ExpectAndReturn(&layers[1], NULL);
    TEST_ASSERT_NULL(scnn_net_init(net));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer layer;
    scnn_layer_connect_Ignore();
    scnn_net_append(net, &layer);

    scnn_layer_init_IgnoreAndReturn(&layer);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_ExpectAndReturn(&layer, &x, &y);
    TEST_ASSERT_EQUAL_PTR(&y, scnn_net_forward(net, &x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer layer[3];
    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &layer[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_IgnoreAndReturn(&layer[i]);
    }
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y[3];
    scnn_layer_forward_ExpectAndReturn(&layer[0], &x, &y[0]);
    scnn_layer_forward_ExpectAndReturn(&layer[1], &y[0], &y[1]);
    scnn_layer_forward_ExpectAndReturn(&layer[2], &y[1], &y[2]);

    TEST_ASSERT_EQUAL_PTR(&y[2], scnn_net_forward(net, &x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_fail_if_net_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer layer;
    scnn_layer_connect_Ignore();
    scnn_net_append(net, &layer);

    scnn_layer_init_IgnoreAndReturn(&layer);
    scnn_net_init(net);

    scnn_dtype x;
    TEST_ASSERT_NULL(scnn_net_forward(NULL, &x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_fail_if_x_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer layer;
    scnn_layer_connect_Ignore();
    scnn_net_append(net, &layer);

    scnn_layer_init_IgnoreAndReturn(&layer);
    scnn_net_init(net);

    scnn_dtype x;
    TEST_ASSERT_NULL(scnn_net_forward(net, NULL));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer layer;
    scnn_layer_connect_Ignore();
    scnn_net_append(net, &layer);

    scnn_layer_init_IgnoreAndReturn(&layer);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    scnn_dtype dy[2];
    scnn_layer_backward_ExpectAndReturn(&layer, &dy[1], &dy[0]);
    TEST_ASSERT_EQUAL_PTR(&dy[0], scnn_net_backward(net, &dy[1]));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[3];
    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &layers[i]);
    }

    scnn_layer_init_IgnoreAndReturn(&layers[0]);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    scnn_dtype dy[4];
    scnn_layer_backward_ExpectAndReturn(&layers[2], &dy[3], &dy[2]);
    scnn_layer_backward_ExpectAndReturn(&layers[1], &dy[2], &dy[1]);
    scnn_layer_backward_ExpectAndReturn(&layers[0], &dy[1], &dy[0]);
    TEST_ASSERT_EQUAL_PTR(&dy[0], scnn_net_backward(net, &dy[3]));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_fail_if_net_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[3];
    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &layers[i]);
    }

    scnn_layer_init_IgnoreAndReturn(&layers[0]);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    scnn_dtype dy[4];
    TEST_ASSERT_NULL(scnn_net_backward(NULL, &dy[3]));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_fail_if_dy_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer layers[3];
    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &layers[i]);
    }

    scnn_layer_init_IgnoreAndReturn(&layers[0]);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    TEST_ASSERT_NULL(scnn_net_backward(net, NULL));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

#if 0
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
#endif
