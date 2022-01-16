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
 */
static void sigmoid_forward(Layer *sigmoid)
{
    const int size = sigmoid->out;
    for (int i = 0; i < size; i++)
    {
        sigmoid->y[i] = 1.0f / (1 + exp(-sigmoid->x[i]));
    }
}

/**
 * @brief allocate Sigmoid layer
 * 
 * @param layer_param layer parameter
 * @return Layer* pointer to layer
 */
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

    layer->forward = sigmoid_forward;

    return layer;

FREE_Y:
    mat_free(&layer->y);
FREE_X:
    mat_free(&layer->x);

    return NULL;
}
