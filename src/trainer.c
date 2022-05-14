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
    float **x,
    float **t,
    const float learning_rate,
    const int epoch,
    const int data_size,
    float (*loss_func)(const float*, const float*, const int))
{
    // indices of learning data
    int *indices = malloc(sizeof(int) * data_size);
    for (int i = 0; i < data_size; i++) {
        indices[i] = i;
    }

    // epoch
    for (int i = 0; i < epoch; i++) {
        shuffle_indices(indices, data_size);

        // training iteration
        for (int j = 0; j < data_size; j++) {
            int index = indices[j];

            net_forward(net, x[index]);

            net_backward(net, t[index]);

            // update network parameters
            for (int n = 0; n < net->size; n++) {
                Layer *layer = net->layers[n];
                layer->update(layer, learning_rate);
            }
        }

        // calculate training loss
        float loss = 0;
        for (int j = 0; j < data_size; j++) {
            net_forward(net, x[j]);
            loss += loss_func(net->output_layer->y, t[j], net->output_layer->y_size);
        }
        loss /= data_size;

        printf("epoch %d: training loss=%f\n", (i + 1), loss);
    }

    FREE_WITH_NULL(&indices);
}
