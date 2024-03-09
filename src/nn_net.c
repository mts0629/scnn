/**
 * @file nn_net.c
 * @brief Network structure
 *
 */
#include "nn_net.h"

#include <stdlib.h>
#include <stdbool.h>

int nn_net_size(const NnNet *net) {
    return net->size;
}

NnLayer *nn_net_layers(NnNet *net) {
    return net->layers;
}

NnLayer *nn_net_input(const NnNet *net) {
    return &net->layers[0];
}
 
NnLayer *nn_net_output(const NnNet *net) {
    if (net->size == 0) {
        return NULL;
    }
    return &net->layers[net->size - 1];
}

NnNet *nn_net_alloc_layers(
    NnNet *net, const int num_layers, NnLayerParams *paramList
) {
    if ((net == NULL) || (num_layers < 1) || (paramList == NULL)) {
        return NULL;
    }

    NnLayer *layers = malloc(sizeof(NnLayer) * num_layers);
    if (layers == NULL) {
        return NULL;
    }

    net->layers = layers;

    // Initialize new layers
    for (int i = 0; i < num_layers; i++) {
        NnLayer *layer = &layers[i];
        layer->batch_size = paramList[i].batch_size;
        layer->in = paramList[i].in;
        layer->out = paramList[i].out;

        // Connect layers
        if (i > 0) {
            nn_layer_connect(&layers[i - 1], &layers[i]);
        }

        layer->x = NULL;
        layer->y = NULL;
        layer->z = NULL;
        layer->w = NULL;
        layer->b = NULL;

        layer->dx = NULL;
        layer->dz = NULL;
        layer->dw = NULL;
        layer->db = NULL;
    }

    net->size = num_layers;

    // Allocate parameters for each layer
    for (int i = 0; i < num_layers; i++) {
        NnLayer *layer = &layers[i];
        if (nn_layer_alloc_params(layer) == NULL) {
            goto FREE_LAYERS;
        }
    }

    return net;

FREE_LAYERS:
    nn_net_free_layers(net);

    return NULL;
}

void nn_net_free_layers(NnNet *net) {
    if ((net == NULL) || (net->layers == NULL)) {
        return;
    }

    for (int i = 0; i < net->size; i++) {
        nn_layer_free_params(&net->layers[i]);
    }

    free(net->layers);
    net->layers = NULL;
}

float *nn_net_forward(NnNet *net, const float *x) {
    if ((net == NULL) || (x == NULL)) {
        return NULL;
    }

    float *in = (float*)x;
    float *out = NULL;
    for (int i = 0; i < net->size; i++) {
        out = nn_layer_forward(&net->layers[i], in);
        in = out;
    }

    return out;
}

float *nn_net_backward(NnNet *net, const float *dy) {
    if ((net == NULL) || (dy == NULL)) {
        return NULL;
    }

    float *din = (float*)dy;
    float *dout = NULL;
    for (int i = (net->size - 1); i >= 0; i--) {
        dout = nn_layer_backward(&net->layers[i], din);
        din = dout;
    }

    return dout;
}

void nn_net_update(NnNet *net, const float learning_rate) {
    for (int i = 0; i < net->size; i++) {
        nn_layer_update(&net->layers[i], learning_rate);
    }
}
