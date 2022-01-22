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
 * @brief allocate layer
 * 
 * @param[in] name layer name
 * @return Layer* pointer to layer
 */
Layer *layer_alloc(const LayerParameter layer_param)
{
    Layer *layer = malloc(sizeof(Layer));
    if (layer == NULL)
    {
        return NULL;
    }

    // initialize basic members
    strncpy(layer->name, layer_param.name, LAYER_NAME_MAX_LENGTH);

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

    return layer;
}

/**
 * @brief deallocate layer
 * 
 * @param[out] layer address of pointer to layer
 */
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
