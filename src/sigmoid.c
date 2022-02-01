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
 * @param sigmoid forwarding layer
 * @param x layer input
 */
static void sigmoid_forward(Layer *sigmoid, const float *x)
{
    sigmoid->x = x;

    const int size = sigmoid->out;
    for (int i = 0; i < size; i++)
    {
        sigmoid->y[i] = 1.0f / (1 + exp(-sigmoid->x[i]));
    }
}

/**
 * @brief backward propagation of Sigmoid layer
 * 
 * @param sigmoid backwarding layer
 * @param dy diff of next layer
 */
static void sigmoid_backward(Layer *sigmoid, const float *dy)
{
    const int size = sigmoid->out;
    for (int i = 0; i < size; i++)
    {
        sigmoid->dx[i] = dy[i] * (1.0f - sigmoid->y[i]) * sigmoid->y[i];
    }
}

Layer *sigmoid_alloc(const LayerParameter layer_param)
{
    if (layer_param.in < 1)
    {
        return NULL;
    }

    Layer *layer = layer_alloc(layer_param);
    if (layer == NULL)
    {
        return NULL;
    }

    layer->in = layer_param.in;

    layer->out = layer_param.in;
    layer->y = mat_alloc(1, layer->out);
    if (layer->y == NULL)
    {
        goto LAYER_FREE;
    }

    layer->dx = mat_alloc(1, layer->in);
    if (layer->dx == NULL)
    {
        goto LAYER_FREE;
    }

    layer->forward = sigmoid_forward;

    layer->backward = sigmoid_backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
