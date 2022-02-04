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
 * @param sigmoid forwarding layer
 * @param x layer input
 */
static void softmax_forward(Layer *softmax, const float *x)
{
    softmax->x = x;

    const int size = softmax->out;

    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += exp(softmax->x[i]);
    }

    for (int i = 0; i < size; i++) {
        softmax->y[i] = exp(softmax->x[i]) / sum;
    }
}

/**
 * @brief backward propagation of Softmax layer
 * 
 * @param softmax backwaoftding layer
 * @param dy diff of output
 */
static void softmax_backward(Layer *softmax, const float *dy)
{
    // backward with cross entropy loss
    for (int i = 0; i < softmax->out; i++) {
        softmax->dx[i] = dy[i];
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

    layer->forward = softmax_forward;

    layer->backward = softmax_backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
