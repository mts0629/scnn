/**
 * @file sigmoid.c
 * @brief Sigmoid layer
 * 
 */
#include "sigmoid.h"

#include <math.h>

#include "data.h"
#include "mat.h"

/**
 * @brief forward propagation of Sigmoid layer
 * 
 * @param self target layer
 * @param x layer input
 */
static void forward(Layer *self, const float *x)
{
    self->x = x;

    for (int i = 0; i < self->x_size; i++) {
        self->y[i] = 1.0f / (1 + exp(-self->x[i]));
    }
}

/**
 * @brief backward propagation of Sigmoid layer
 * 
 * @param self target layer
 * @param dy diff of next layer
 */
static void backward(Layer *self, const float *dy)
{
    for (int i = 0; i < self->y_size; i++) {
        self->dx[i] = dy[i] * (1.0f - self->y[i]) * self->y[i];
    }
}

Layer *sigmoid_layer(const LayerParameter layer_param)
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
