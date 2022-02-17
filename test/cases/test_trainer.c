/**
 * @file test_train.c
 * @brief unit test of trainer.c
 * 
 */
#include "layers.h"
#include "mat.h"
#include "trainer.h"
#include "random.h"
#include "loss.h"

#include "unity_fixture.h"

TEST_GROUP(trainer);

TEST_SETUP(trainer)
{}

TEST_TEAR_DOWN(trainer)
{}

TEST(trainer, train_sgd)
{
    rand_seed(0);

    Net *net = net_create(
        4,
        (Layer*[]){
            fc_layer((LayerParameter){ .in=2, .out=10 }),
            sigmoid_layer((LayerParameter){ .in=10 }),
            fc_layer((LayerParameter){ .in=10, .out=1 }),
            sigmoid_layer((LayerParameter){ .in=1 })
        }
    );

    mat_randomize_norm(net->layers[0]->w, (net->layers[0]->x_dim[1] * net->layers[0]->y_dim[1]), 0, 1);
    mat_randomize_norm(net->layers[0]->b, net->layers[0]->y_dim[1], 0, 1);

    mat_randomize_norm(net->layers[2]->w, (net->layers[2]->x_dim[1] * net->layers[2]->y_dim[1]), 0, 1);
    mat_randomize_norm(net->layers[2]->b, net->layers[2]->y_dim[1], 0, 1);

    float *x[] = {
        (float[2]){ 0, 0 },
        (float[2]){ 0, 1 },
        (float[2]){ 1, 0 },
        (float[2]){ 1, 1 },
    };

    float *t[] = {
        (float[1]){ 0 },
        (float[1]){ 1 },
        (float[1]){ 1 },
        (float[1]){ 0 }
    };

    printf("\n");

    train_sgd(net, x, t, 0.3, 1000, 4, 4, mean_squared_error);

    for (int i = 0; i < 4; i++) {
        net_forward(net, x[i]);
        // prediction
        int pred  = (1 ? (net->output_layer->y[0] > 0.5) : 0);
        TEST_ASSERT_EQUAL_INT(t[i][0], pred);
    }

    net_free(&net);
}
