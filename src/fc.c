/**
 * @file fc.c
 * @brief Fully connected layer
 * 
 */
#include "fc.h"

#include "data.h"
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
    fdata_copy(dy, self->b_size, self->db);
}

Layer *fc_layer(const LayerParameter layer_param)
{
    if ((layer_param.in < 1) || (layer_param.out < 1)) {
        return NULL;
    }

    Layer *layer = layer_alloc();
    if (layer == NULL) {
        return NULL;
    }

    int x_size = 1 * layer_param.in * 1 * 1;
    SET_DIM(layer->x_dim, 1, layer_param.in, 1, 1);
    layer->x_size = x_size;

    int y_size = 1 * layer_param.out * 1 * 1;
    SET_DIM(layer->y_dim, 1, layer_param.out, 1, 1);
    layer->y_size = y_size;

    int w_size = layer_param.out * layer_param.in * 1 * 1;
    SET_DIM(layer->w_dim, layer_param.out, layer_param.in, 1, 1);
    layer->w_size = w_size;

    int b_size = y_size;
    SET_DIM(layer->b_dim, 1, layer_param.out, 1, 1);
    layer->b_size = b_size;

    layer->y = fdata_alloc(y_size);
    if (layer->y == NULL) {
        goto LAYER_FREE;
    }

    layer->w = fdata_alloc(w_size);
    if (layer->w == NULL) {
        goto LAYER_FREE;
    }

    layer->b = fdata_alloc(b_size);
    if (layer->b == NULL) {
        goto LAYER_FREE;
    }

    layer->dx = fdata_alloc(x_size);
    if (layer->dx == NULL) {
        goto LAYER_FREE;
    }

    layer->dw = fdata_alloc(w_size);
    if (layer->dw == NULL) {
        goto LAYER_FREE;
    }

    layer->db = fdata_alloc(b_size);
    if (layer->db == NULL) {
        goto LAYER_FREE;
    }

    layer->forward  = forward;
    layer->backward = backward;

    return layer;

LAYER_FREE:
    layer_free(&layer);

    return NULL;
}
