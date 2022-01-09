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
Layer *layer_alloc(const char *name)
{
    Layer *layer = malloc(sizeof(Layer));
    if (layer == NULL)
    {
        return NULL;
    }

    // initialize members
    strncpy(layer->name, name, LAYER_NAME_MAX_LENGTH);

    layer->x = NULL;
    layer->y = NULL;

    layer->prev = NULL;
    layer-> next = NULL;

    layer->forward = NULL;

    return layer;
}

/**
 * @brief deallocate layer
 * 
 * @param[out] layer address of pointer to layer
 */
void layer_free(Layer **layer)
{
    mat_free(&(*layer)->x);
    mat_free(&(*layer)->y);

    (*layer)->prev = NULL;
    (*layer)-> next = NULL;

    free(*layer);
    *layer = NULL;
}
