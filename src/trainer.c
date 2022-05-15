/**
 * @file trainer.c
 * @brief network training operations
 * 
 */
#include "trainer.h"

#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "mat.h"
#include "random.h"

/**
 * @brief shuffle data indices with specified batch size for training
 * 
 * @param[in] indices array of indices
 * @param[out] shuffled array of shuffled indices
 * @param[in] data_size num of training data
 * @param[in] batch_size training batch size
 */
static void shuffle_indices(int *indices, const int data_size)
{
    // swap randomly specified indices
    // iterate with data size
    for (int i = 0; i < data_size; i++) {
        int idx0 = rand_xorshift() % data_size;
        int idx1 = rand_xorshift() % data_size;

        int tmp = indices[idx1];
        indices[idx1] = indices[idx0];
        indices[idx0] = tmp;
    }
}

void train_sgd(
    Net *net,
    float **train_x,
    float **train_t,
    float **test_x,
    float **test_t,
    const float learning_rate,
    const int epoch,
    const int train_data_size,
    const int test_data_size,
    float (*loss_func)(const float*, const float*, const int))
{
    // indices of learning data
    int *indices = malloc(sizeof(int) * train_data_size);
    for (int i = 0; i < train_data_size; i++) {
        indices[i] = i;
    }

    // epoch
    for (int i = 0; i < epoch; i++) {
        shuffle_indices(indices, train_data_size);

        // training iteration
        for (int j = 0; j < train_data_size; j++) {
            int index = indices[j];

            net_forward(net, train_x[index]);

            net_backward(net, train_t[index]);

            // update network parameters
            for (int n = 0; n < net->size; n++) {
                Layer *layer = net->layers[n];
                layer->update(layer, learning_rate);
            }
        }

        printf("epoch %d: ", (i + 1));

        // calculate training loss
        float train_loss = 0;
        for (int j = 0; j < train_data_size; j++) {
            net_forward(net, train_x[j]);
            train_loss += loss_func(net->output_layer->y, train_t[j], net->output_layer->y_size);
        }
        train_loss /= train_data_size;

        printf("training loss=%f", train_loss);

        if (test_x != NULL && test_t != NULL) {
            // calculate test loss
            float test_loss = 0;
            for (int j = 0; j < test_data_size; j++) {
                net_forward(net, test_x[j]);
                test_loss += loss_func(net->output_layer->y, test_t[j], net->output_layer->y_size);
            }
            test_loss /= test_data_size;
        }

        printf("\n");
    }

    FREE_WITH_NULL(&indices);
}
