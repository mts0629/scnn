/**
 * @file layer.c
 * @brief basic layer struct and operations
 * 
 */

#include "layer.h"

#include <stdlib.h>
#include <math.h>

#include "data.h"
#include "random.h"
#include "util.h"
#include "mat.h"

/**
 * @brief initialize layer parameters
 * 
 * @param[in,out] self target layer
 */
static void init_params(Layer *self)
{
    // temporally, implement with Xavier initialization
    if (self->w != NULL) {
        float scale  = 1.0f / sqrt(1.0f / self->x_size);
        for (int i = 0; i < self->w_size; i++) {
            self->w[i] = rand_norm(0, 1) * scale;
        }
    }
    if (self->b != NULL) {
        for (int i = 0; i < self->b_size; i++) {
            self->b[i] = 0;
        }
    }
}

/**
 * @brief update layer parameters with specified learning rate
 * 
 * @param[in,out] self target layer
 * @param[in] learning_rate learning rate
 */
static void update(Layer *self, const float learning_rate)
{
    // update weights
    if (self->w != NULL) {
        for (int i = 0; i < self->w_size; i++) {
            self->w[i] -= learning_rate * self->dw[i];
        }
    }

    // update biases
    if (self->b != NULL) {
        for (int i = 0; i < self->b_size; i++) {
            self->b[i] -= learning_rate * self->db[i];
        }
    }
}

Layer *layer_alloc(void)
{
    Layer *layer = malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    // initialize basic members
    layer->id = -1;

    layer->x = NULL;
    layer->x_size = 0;

    layer->y = NULL;
    layer->y_size = 0;

    layer->w = NULL;
    layer->w_size = 0;

    layer->b = NULL;
    layer->b_size = 0;

    for (int i = 0; i < N_DIM; i++) {
        layer->x_dim[i] = 0;
        layer->y_dim[i] = 0;
        layer->w_dim[i] = 0;
        layer->b_dim[i] = 0;
    }

    layer->dx = NULL;
    layer->dw = NULL;
    layer->db = NULL;

    layer->prev_id = -1;
    layer->next_id = -1;

    layer->forward = NULL;
    layer->backward = NULL;

    layer->init_params = init_params;

    layer->update = update;

    return layer;
}

void layer_free(Layer **layer)
{
    FREE_WITH_NULL(&(*layer)->y);

    FREE_WITH_NULL(&(*layer)->w);
    FREE_WITH_NULL(&(*layer)->b);

    FREE_WITH_NULL(&(*layer)->dx);
    FREE_WITH_NULL(&(*layer)->dw);
    FREE_WITH_NULL(&(*layer)->db);

    FREE_WITH_NULL(layer);
}
