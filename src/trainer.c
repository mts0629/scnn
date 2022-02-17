/**
 * @file trainer.c
 * @brief network training operations
 * 
 */
#include "trainer.h"

#include <stdio.h>
#include <stdlib.h>

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
static void shuffle_indices(int *indices, int *shuffled, const int data_size, const int batch_size)
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

    // get shuffled indices
    for (int i = 0; i < batch_size; i++) {
        // naive implementation: get specified size of indices from head
        shuffled[i] = indices[i];
    }
}

void train_sgd(Net *net,
    float **x, float **t,
    const float learning_rate,
    const int epoch,
    const int data_size, const int batch_size,
    float (*loss_func)(const float*, const float*, const int))
{
    // allocate dy
    float *dy = mat_alloc(1, net->output_layer->y_size);

    // create indices of learning data
    int *indices = malloc(sizeof(int) * data_size);

    // create indices randomly selected
    int *batch_indices = malloc(sizeof(int) * batch_size);

    for (int i = 0; i < epoch; i++) {
        float loss = 0;

        shuffle_indices(indices, batch_indices, data_size, batch_size);

        for (int j = 0; j < batch_size; j++) {
            int index = batch_indices[j];

            net_forward(net, x[index]);

            loss += loss_func(net->output_layer->y, t[index], net->output_layer->y_size);

            mat_sub(net->output_layer->y, t[index], dy, 1, net->output_layer->y_size);

            net_backward(net, dy);

            // update network parameters
            for (int n = 0; n < net->length; n++) {
                Layer *layer = net->layers[n];
                layer->update(layer, learning_rate);
            }
        }

        printf("epoch [%d] loss=%f\n", i, loss);
    }

    free(indices);
    indices = NULL;

    free(batch_indices);
    batch_indices = NULL;
}
