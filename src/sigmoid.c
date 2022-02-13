/**
 * @file sigmoid.c
 * @brief Sigmoid layer
 * 
 */
#include "sigmoid.h"

#include <math.h>

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

Layer *sigmoid_alloc(const LayerParameter layer_param)
{
    if (layer_param.in < 1) {
        return NULL;
    }

    Layer *layer = layer_alloc();
    if (layer == NULL) {
        return NULL;
    }

    layer->x_dim[0] = 1;
    layer->x_dim[1] = layer_param.in;
    layer->x_dim[2] = 1;
    layer->x_dim[3] = 1;
    layer->x_size = layer->x_dim[0] * layer->x_dim[1] * layer->x_dim[2] * layer->x_dim[3];

    layer->y_dim[0] = 1;
    layer->y_dim[1] = layer_param.in;
    layer->y_dim[2] = 1;
    layer->y_dim[3] = 1;
    layer->y_size = layer->y_dim[0] * layer->y_dim[1] * layer->y_dim[2] * layer->y_dim[3];

    layer->y = mat_alloc(1, layer_param.in);
    if (layer->y == NULL) {
        goto LAYER_FREE;
    }

    layer->dx = mat_alloc(1, layer_param.in);
    if (layer->dx == NULL) {
        goto LAYER_FREE;
    }

    layer->forward = forward;

    layer->backward = backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
