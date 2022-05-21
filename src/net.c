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
#include "mat.h"

Net *net_alloc(void)
{
    Net *net = malloc(sizeof(Net));
    if (net == NULL) {
        return NULL;
    }

    // initialize member
    net->size = 0;

    for (int i = 0; i < NET_LAYER_MAX_SIZE; i++) {
        net->layers[i] = NULL;
    }

    net->input_layer  = NULL;
    net->output_layer = NULL;

    return net;
}

Net *net_create(const int size, Layer *layers[])
{
    if ((size < 1) || (size > NET_LAYER_MAX_SIZE) || (layers == NULL)) {
        return NULL;
    }

    Net *net = net_alloc();
    if (net == NULL) {
        return NULL;
    }

    net->size = 0;

    for (int i = 0; i < size; i++) {
        if (net_append(net, layers[i]) == NULL) {
            goto NET_FREE;
        }
    }

    return net;

NET_FREE:
    net_free(&net);

    return NULL;
}

Net *net_append(Net *net, Layer *layer)
{
    if ((net == NULL) || (layer == NULL)) {
        return NULL;
    }

    int id = net->size;

    layer->id      = id;
    layer->next_id = -1;

    // append layer to tail of the network layers
    if (id == 0) {
        // first layer
        layer->x       = NULL;
        layer->prev_id = -1;
    } else {
        Layer *prev_layer = net->layers[id - 1];

        layer->x       = prev_layer->y;
        layer->prev_id = prev_layer->id;

        prev_layer->next_id = id;
    }
    net->layers[id] = layer;

    net->input_layer  = net->layers[0];
    net->output_layer = net->layers[id];

    net->size++;

    return net;
}

void net_init_layer_params(Net *net)
{
    for (int i = 0; i < net->size; i++) {
        net->layers[i]->init_params(net->layers[i]);
    }
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

void net_backward(Net *net, const float *t)
{
    // calculate diff at the last layer
    float *dy = malloc(sizeof(float) * net->output_layer->y_size);

    mat_sub(net->output_layer->y, t, dy, 1, net->output_layer->y_size);

    net->output_layer->backward(net->output_layer, dy);

    Layer *next_layer = net->output_layer;
    Layer *layer = net->layers[next_layer->prev_id];

    // backwarding
    while (true) {
        layer->backward(layer, next_layer->dx);
        int prev_id = layer->prev_id;
        if (prev_id < 0) {
            break;
        }
        next_layer = layer;
        layer = net->layers[prev_id];
    }

    FREE_WITH_NULL(&dy);
}

void net_free(Net **net)
{
    if (*net == NULL) {
        return;
    }

    Layer *layer = (*net)->layers[0];
    if (layer == NULL) {
        goto NET_FREE;
    }

    while (true) {
        int next_id = layer->next_id;
        layer_free(&layer);
        if (next_id < 0) {
            break;
        }
        layer = (*net)->layers[next_id];
    }

NET_FREE:
    FREE_WITH_NULL(net);
}
