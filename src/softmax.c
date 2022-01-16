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
 */
static void softmax_forward(Layer *softmax)
{
    const int size = softmax->out;

    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += exp(softmax->x[i]);
    }

    for (int i = 0; i < size; i++)
    {
        softmax->y[i] = exp(softmax->x[i]) / sum;
    }
}

/**
 * @brief allocate Softmax layer
 * 
 * @param layer_param layer parameter
 * @return Layer* pointer to layer
 */
Layer *softmax_alloc(const LayerParameter layer_param)
{
    Layer *layer = layer_alloc(layer_param);
    if (layer == NULL)
    {
        return NULL;
    }

    layer->in = layer_param.in;
    layer->x = mat_alloc(1, layer->in);
    if (layer->x == NULL)
    {
        goto FREE_X;
    }

    layer->out = layer_param.in;
    layer->y = mat_alloc(1, layer->out);
    if (layer->y == NULL)
    {
        goto FREE_Y;
    }

    layer->forward = softmax_forward;

    return layer;

FREE_Y:
    mat_free(&layer->y);
FREE_X:
    mat_free(&layer->x);

    return NULL;
}
