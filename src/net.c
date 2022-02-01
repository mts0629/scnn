/**
 * @file net.c
 * @brief network structure
 * 
 */
#include "net.h"

#include <stdlib.h>
#include <string.h>

#include "mat.h"

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

    net->num_layers = length;

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
 * @brief forward propagation of network
 * 
 */
void net_forward(Net *net, const float *x)
{
    net->layers[0]->x = x;

    for (int i = 0; ; i++)
    {
        net->layers[i]->forward(net->layers[i], net->layers[i]->x);

        if (net->layers[i]->next == NULL)
        {
            break;
        }
    }
}

/**
 * @brief backward propagation of network
 * 
 */
void net_backward(Net *net, const float *t)
{
    // get diff of network output
    int out = net->layers[net->num_layers - 1]->out;

    float *dy = mat_alloc(1, out);
    float *y = net->layers[net->num_layers - 1]->y;

    mat_sub(y, t, dy, 1, out);

    net->layers[net->num_layers - 1]->backward(net->layers[net->num_layers - 1], dy);

    for (int i = (net->num_layers - 2); ; i--)
    {
        net->layers[i]->backward(net->layers[i], net->layers[i + 1]->dx);

        if (net->layers[i]->prev == NULL)
        {
            break;
        }
    }

    mat_free(&dy);
}

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
