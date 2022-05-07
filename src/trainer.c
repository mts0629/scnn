/**
 * @file trainer.c
 * @brief network training operations
 * 
 */
#include "trainer.h"

#include <stdio.h>
#include <stdlib.h>

#include "data.h"
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
    // initialize indices
    for (int i = 0; i < data_size; i++) {
        indices[i] = i;
    }

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

#if 0
static void get_batch_indices(int *indices, int *batch_indices, const int data_size, const int batch_size)
{
    // get shuffled indices
    for (int i = 0; i < batch_size; i++) {
        // naive implementation: get specified size of indices from head
        batch_indices[i] = indices[i];
    }
}
#endif

void train_sgd(
    Net *net,
    float **x,
    float **t,
    const float learning_rate,
    const int epoch,
    const int data_size,
    float (*loss_func)(const float*, const float*, const int))
{
    // allocate dy
    float *dy = fdata_alloc(net->output_layer->y_size);

    // create indices of learning data
    int *indices = malloc(sizeof(int) * data_size);

    for (int i = 0; i < epoch; i++) {
        float loss = 0;

        shuffle_indices(indices, data_size);

        for (int j = 0; j < data_size; j++) {
            int index = indices[j];

            net_forward(net, x[index]);

            mat_sub(net->output_layer->y, t[index], dy, 1, net->output_layer->y_size);

            net_backward(net, dy);

            // update network parameters
            for (int n = 0; n < net->size; n++) {
                Layer *layer = net->layers[n];
                layer->update(layer, learning_rate);
            }

            net_forward(net, x[index]);

            loss += loss_func(net->output_layer->y, t[index], net->output_layer->y_size);
        }

        // averaging with data size
        loss /= data_size;

        //printf("epoch [%d] loss=%f\n", i, loss);
    }

    FREE_WITH_NULL(&dy);

    FREE_WITH_NULL(&indices);
}
