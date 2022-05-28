/**
 * @file softmax.c
 * @brief Softmax layer
 * 
 */
#include "softmax.h"

#include <math.h>

#include "data.h"
#include "mat.h"

/**
 * @brief forward propagation of Softmax layer
 * 
 * @param self target layer
 * @param x layer input
 */
static void forward(Layer *self, const float *x)
{
    self->x = x;

    float sum = 0;
    for (int i = 0; i < self->x_size; i++) {
        sum += exp(self->x[i]);
    }

    for (int i = 0; i < self->x_size; i++) {
        self->y[i] = exp(self->x[i]) / sum;
    }
}

/**
 * @brief backward propagation of Softmax layer
 * 
 * @param self backwaoftding layer
 * @param dy diff of output
 */
static void backward(Layer *self, const float *dy)
{
    // backward with cross entropy loss
    for (int i = 0; i < self->y_size; i++) {
        self->dx[i] = dy[i];
    }
}

Layer *softmax_layer(const LayerParameter layer_param)
{
    if (layer_param.in < 1) {
        return NULL;
    }

    Layer *layer = layer_alloc();
    if (layer == NULL) {
        return NULL;
    }

    int x_size = 1 * layer_param.in * 1 * 1;
    SET_DIM(layer->x_dim, 1, layer_param.in, 1, 1);
    layer->x_size = x_size;

    int y_size = 1 * layer_param.in * 1 * 1;
    SET_DIM(layer->y_dim, 1, layer_param.in, 1, 1);
    layer->y_size = y_size;

    layer->y = fdata_alloc(y_size);
    if (layer->y == NULL) {
        goto LAYER_FREE;
    }

    layer->dx = fdata_alloc(x_size);
    if (layer->dx == NULL) {
        goto LAYER_FREE;
    }

    layer->forward  = forward;
    layer->backward = backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
