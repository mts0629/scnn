/**
 * @file net.c
 * @brief network structure
 * 
 */
#include "net.h"

#include <stdlib.h>
#include <stdbool.h>

#include "data.h"
#include "util.h"

Net *net_create(const int size, Layer *layers[])
{
    if ((size < 1) || (layers == NULL)) {
        return NULL;
    }

    Net *net = malloc(sizeof(Net));
    if (net == NULL) {
        return NULL;
    }

    net->layers = malloc(sizeof(Layer*) * size);
    if (net->layers == NULL) {
        FREE_WITH_NULL(net);
        return NULL;
    }

    net->size = size;

    int id = 0;
    Layer *prev_layer = NULL;

    for (int i = 0; i < size; i++) {
        net->layers[i] = layers[i];

        net->layers[i]->id = id;

        if (prev_layer != NULL) {
            net->layers[i]->x = prev_layer->y;
            net->layers[i]->prev_id = prev_layer->id;
            net->layers[i]->next_id = -1;

            prev_layer->next_id = id;
        }

        prev_layer = net->layers[i];

        id++;
    }

    net->input_layer = net->layers[0];
    net->output_layer = net->layers[size - 1];

    return net;
}

void net_forward(Net *net, const float *x)
{
    Layer *layer = net->layers[0];

    layer->x = x;

    while (true) {
        layer->forward(layer, layer->x);
        int next_id = layer->next_id;
        if (next_id < 0) {
            break;
        }
        layer = net->layers[next_id];
    }
}

void net_backward(Net *net, const float *dy)
{
    net->output_layer->backward(net->output_layer, dy);

    Layer *next_layer = net->output_layer;
    Layer *layer = net->layers[next_layer->prev_id];

    while (true) {
        layer->backward(layer, next_layer->dx);
        int prev_id = layer->prev_id;
        if (prev_id < 0) {
            break;
        }
        next_layer = layer;
        layer = net->layers[prev_id];
    }
}

void net_free(Net **net)
{
    Layer *layer = (*net)->layers[0];

    while (true) {
        int next_id = layer->next_id;
        layer_free(&layer);
        if (next_id < 0) {
            break;
        }
        layer = (*net)->layers[next_id];
    }

    FREE_WITH_NULL(net);
}
