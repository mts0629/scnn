/**
 * @file test_train.c
 * @brief unit test of trainer.c
 * 
 */
#include "data.h"
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

    net_init_layer_params(net);

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

    int correct = 0;
    for (int i = 0; i < 4; i++) {
        net_forward(net, x[i]);
        int pred = (1 ? (net->output_layer->y[0] > 0.5) : 0);
        if (pred == t[i][0]) {
            correct++;
        }
    }

    float prev_acc = (float)correct / 4;

    train_sgd(net, x, t, NULL, NULL, 0.1, 200, 4, 0, mean_squared_error);

    correct = 0;
    for (int i = 0; i < 4; i++) {
        net_forward(net, x[i]);
        int pred = (1 ? (net->output_layer->y[0] > 0.5) : 0);
        if (pred == t[i][0]) {
            correct++;
        }
    }

    float acc = (float)correct / 4;

    printf("accuracy: prev=%f, now=%f\n", prev_acc, acc);

    TEST_ASSERT((acc >= prev_acc));

    net_free(&net);
}
