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
 * @param x layer input
 */
static void fc_forward(Layer *fc, const float *x)
{
    fc->x = x;

    mat_mul(fc->x, fc->w, fc->y, 1, fc->in, fc->out);
    mat_add(fc->y, fc->b, fc->y, 1, fc->out);
}

/**
 * @brief backward propagation of Fully connected layer
 * 
 * @param fc backwarding layer
 * @param dy diff of next layer
 */
static void fc_backward(Layer *fc, const float *dy)
{
    mat_mul_trans_b(dy, fc->w, fc->dx, 1, fc->out, fc->in);

    mat_mul_trans_a(fc->x, dy, fc->dw, 1, fc->in, fc->out);

    mat_copy(dy, 1, fc->out, fc->db);
}

Layer *fc_alloc(const LayerParameter layer_param)
{
    if ((layer_param.in < 1) || (layer_param.out < 1)) {
        return NULL;
    }

    Layer *layer = layer_alloc(layer_param);
    if (layer == NULL) {
        return NULL;
    }

    layer->in = layer_param.in;

    layer->out = layer_param.out;
    layer->y = mat_alloc(1, layer->out);
    if (layer->y == NULL) {
        goto LAYER_FREE;
    }

    layer->w = mat_alloc(layer->in, layer->out);
    if (layer->w == NULL) {
        goto LAYER_FREE;
    }

    layer->b = mat_alloc(1, layer->out);
    if (layer->b == NULL) {
        goto LAYER_FREE;
    }

    layer->dx = mat_alloc(1, layer->in);
    if (layer->dx == NULL) {
        goto LAYER_FREE;
    }

    layer->dw = mat_alloc(layer->in, layer->out);
    if (layer->dw == NULL) {
        goto LAYER_FREE;
    }

    layer->db = mat_alloc(1, layer->out);
    if (layer->db == NULL) {
        goto LAYER_FREE;
    }

    layer->forward = fc_forward;

    layer->backward = fc_backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
