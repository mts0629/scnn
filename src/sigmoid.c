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

    const int size = self->out;
    for (int i = 0; i < size; i++) {
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
    const int size = self->out;
    for (int i = 0; i < size; i++) {
        self->dx[i] = dy[i] * (1.0f - self->y[i]) * self->y[i];
    }
}

Layer *sigmoid_alloc(const LayerParameter layer_param)
{
    if (layer_param.in < 1) {
        return NULL;
    }

    Layer *layer = layer_alloc(layer_param);
    if (layer == NULL) {
        return NULL;
    }

    layer->in = layer_param.in;

    layer->out = layer_param.in;
    layer->y = mat_alloc(1, layer->out);
    if (layer->y == NULL) {
        goto LAYER_FREE;
    }

    layer->dx = mat_alloc(1, layer->in);
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
