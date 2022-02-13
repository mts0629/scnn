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

    // y = Wx + b
    mat_mul(self->x, self->w, self->y, 1, self->x_dim[1], self->y_dim[1]);
    mat_add(self->y, self->b, self->y, 1, self->y_dim[1]);
}

/**
 * @brief backward propagation of Fully connected layer
 * 
 * @param self target layer
 * @param dy diff of next layer
 */
static void backward(Layer *self, const float *dy)
{
    // dy = Wx^T
    mat_mul_trans_b(dy, self->w, self->dx, 1, self->y_dim[1], self->x_dim[1]);
    // dW = x^T dy
    mat_mul_trans_a(self->x, dy, self->dw, 1, self->x_dim[1], self->y_dim[1]);
    // db = dy
    mat_copy(dy, 1, self->b_size, self->db);
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

    layer->x_dim[0] = 1;
    layer->x_dim[1] = layer_param.in;
    layer->x_dim[2] = 1;
    layer->x_dim[3] = 1;
    layer->x_size = layer->x_dim[0] * layer->x_dim[1] * layer->x_dim[2] * layer->x_dim[3];

    layer->y_dim[0] = 1;
    layer->y_dim[1] = layer_param.out;
    layer->y_dim[2] = 1;
    layer->y_dim[3] = 1;
    layer->y_size = layer->y_dim[0] * layer->y_dim[1] * layer->y_dim[2] * layer->y_dim[3];

    layer->y = mat_alloc(1, layer_param.out);
    if (layer->y == NULL) {
        goto LAYER_FREE;
    }

    layer->w_dim[0] = layer_param.out;
    layer->w_dim[1] = layer_param.in;
    layer->w_dim[2] = 1;
    layer->w_dim[3] = 1;
    layer->w_size = layer->w_dim[0] * layer->w_dim[1] * layer->w_dim[2] * layer->w_dim[3];

    layer->w = mat_alloc(layer_param.in, layer_param.out);
    if (layer->w == NULL) {
        goto LAYER_FREE;
    }

    layer->b_dim[0] = 1;
    layer->b_dim[1] = layer_param.out;
    layer->b_dim[2] = 1;
    layer->b_dim[3] = 1;
    layer->b_size = layer->b_dim[0] * layer->b_dim[1] * layer->b_dim[2] * layer->b_dim[3];

    layer->b = mat_alloc(1, layer_param.out);
    if (layer->b == NULL) {
        goto LAYER_FREE;
    }

    layer->dx = mat_alloc(1, layer_param.in);
    if (layer->dx == NULL) {
        goto LAYER_FREE;
    }

    layer->dw = mat_alloc(layer_param.in, layer_param.out);
    if (layer->dw == NULL) {
        goto LAYER_FREE;
    }

    layer->db = mat_alloc(1, layer_param.out);
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
