/**
 * @file layer.c
 * @brief basic layer struct and operations
 * 
 */

#include "layer.h"

#include <stdlib.h>
#include <string.h>

#include "mat.h"

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
    layer->x = NULL;

    layer->y = NULL;

    layer->w = NULL;
    layer->b = NULL;

    layer->dx = NULL;
    layer->dw = NULL;
    layer->db = NULL;

    layer->prev = NULL;
    layer->next = NULL;

    layer->forward = NULL;
    layer->backward = NULL;

    layer->update = update;

    return layer;
}

void layer_free(Layer **layer)
{
    mat_free(&(*layer)->y);

    mat_free(&(*layer)->w);
    mat_free(&(*layer)->b);

    mat_free(&(*layer)->dx);
    mat_free(&(*layer)->dw);
    mat_free(&(*layer)->db);

    (*layer)->prev = NULL;
    (*layer)->next = NULL;

    (*layer)->forward = NULL;
    (*layer)->backward = NULL;

    free(*layer);
    *layer = NULL;
}
