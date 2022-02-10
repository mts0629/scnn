/**
 * @file softmax.c
 * @brief Softmax layer
 * 
 */
#include "softmax.h"

#include <math.h>

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

    const int size = self->out;

    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += exp(self->x[i]);
    }

    for (int i = 0; i < size; i++) {
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
    for (int i = 0; i < self->out; i++) {
        self->dx[i] = dy[i];
    }
}

Layer *softmax_alloc(const LayerParameter layer_param)
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
