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
 * @param self target layer
 * @param x layer input
 */
static void forward(Layer *self, const float *x)
{
    self->x = x;

    mat_mul(self->x, self->w, self->y, 1, self->in, self->out);
    mat_add(self->y, self->b, self->y, 1, self->out);
}

/**
 * @brief backward propagation of Fully connected layer
 * 
 * @param self target layer
 * @param dy diff of next layer
 */
static void backward(Layer *self, const float *dy)
{
    mat_mul_trans_b(dy, self->w, self->dx, 1, self->out, self->in);

    mat_mul_trans_a(self->x, dy, self->dw, 1, self->in, self->out);

    mat_copy(dy, 1, self->out, self->db);
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

    layer->forward = forward;

    layer->backward = backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
