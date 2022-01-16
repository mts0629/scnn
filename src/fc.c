/**
 * @file fc.c
 * @brief Fully connected layer
 * 
 */
#include "fc.h"

#include "mat.h"

/**
 * @brief forward propagation of Fully connected layer
 * 
 * @param fc forwarding layer
 */
static void fc_forward(Layer *fc)
{
    mat_mul(fc->x, fc->w, fc->y, 1, fc->in, fc->out);
    mat_add(fc->y, fc->b, fc->y, 1, fc->out);
}

/**
 * @brief allocate Fully connected layer
 * 
 * @param layer_param layer parameter
 * @return Layer* pointer to layer
 */
Layer *fc_alloc(const LayerParameter layer_param)
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

    layer->out = layer_param.out;
    layer->y = mat_alloc(1, layer->out);
    if (layer->y == NULL)
    {
        goto FREE_Y;
    }

    layer->w = mat_alloc(layer->in, layer->out);
    if (layer->w == NULL)
    {
        goto FREE_W;
    }

    layer->b = mat_alloc(1, layer->out);
    if (layer->b == NULL)
    {
        goto FREE_B;
    }

    layer->forward = fc_forward;

    return layer;

FREE_B:
    mat_free(&layer->b);
FREE_W:
    mat_free(&layer->w);
FREE_Y:
    mat_free(&layer->y);
FREE_X:
    mat_free(&layer->x);

    return NULL;
}