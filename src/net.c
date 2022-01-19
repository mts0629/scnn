/**
 * @file net.c
 * @brief network structure
 * 
 */
#include "net.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief create network
 * 
 */
Net *net_create(const char *name, const int length, Layer **layers)
{
    if (length < 1)
    {
        return NULL;
    }

    if (layers == NULL)
    {
        return NULL;
    }

    Net *net = malloc(sizeof(Net));
    if (net == NULL)
    {
        return NULL;
    }

    strncpy(net->name, name, NET_NAME_MAX_LENGTH);

    net->layers = malloc(sizeof(Layer*) * length);
    if (net->layers == NULL)
    {
        free(net);
        net = NULL;

        return NULL;
    }

    net->layers[0] = layers[0];

    for (int i = 1; i < length; i++)
    {
        net->layers[i] = layers[i];

        net->layers[i - 1]->next = net->layers[i];

        net->layers[i]->x = net->layers[i - 1]->y;
        net->layers[i]->prev = net->layers[i - 1];
    }

    return net;
}

/**
 * @brief deallocate network
 * 
 */
void net_free(Net **net)
{
    Layer *layer = (*net)->layers[0];

    while (layer != NULL)
    {
        Layer *next = layer->next;
        layer_free(&layer);
        layer = next;
    }

    free(*net);
    *net = NULL;
}
