/**
 * @file scnn_net.c
 * @brief Network structure
 * 
 */
#include <stdlib.h>
#include <stdbool.h>

#include "scnn_net.h"

scnn_net *scnn_net_alloc(void)
{
    scnn_net *net = malloc(sizeof(scnn_net));
    if (net == NULL) {
        return NULL;
    }

    net->size = 0;

    net->batch_size = 1;

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        net->layers[i] = NULL;
    }

    net->input  = NULL;
    net->output = NULL;

    return net;
}

scnn_net *scnn_net_append(scnn_net *net, scnn_layer *layer)
{
    if ((net == NULL) || (layer == NULL)) {
        return NULL;
    }

    if (net->size >= SCNN_NET_MAX_SIZE) {
        return NULL;
    }

    // Set a layer ID with an index
    layer->params.id      = net->size;

    layer->params.prev_id = -1;
    layer->params.next_id = -1;

    scnn_layer *last_layer = net->output;

    // Set layer ID of the current network output as a previous layer
    if (last_layer != NULL) {
        last_layer->params.next_id = layer->params.id;
        layer->params.prev_id      = last_layer->params.id;
    }

    // Append the current layer to tail of the network layers
    net->layers[net->size] = layer;

    // Set the first layer as a network input
    if (net->size == 0) {
        net->input = layer;
    }
    // And the last layer as a network output
    net->output = layer;

    net->size++;

    return net;
}

scnn_net *scnn_net_init(scnn_net *net)
{
    if (net == NULL) {
        return NULL;
    }

    if (net->size == 0) {
        return NULL;
    }

    // initilaize the first layer
    if (net->layers[0]->init(net->layers[0]) == NULL) {
        return NULL;
    }

    // initialize precedding layers
    for (int i = 1; i < net->size; i++) {
        // set an output shape of the previous layer as that of the next input
        for (int j = 0; j < SCNN_MAT_DIM; j++) {
            net->layers[i]->params.in_shape[j] = net->layers[i - 1]->y->shape[j];
        }

        if (net->layers[i]->init(net->layers[i]) == NULL) {
            return NULL;
        }
    }

    return net;
}

void scnn_net_forward(scnn_net *net, scnn_dtype *x)
{
    if ((net == NULL) || (x == NULL)) {
        return;
    }

    scnn_dtype  *in = x;
    for (int i = 0; i < net->size; i++) {
        net->layers[i]->forward(net->layers[i], in);
        in = net->layers[i]->y->data;
    }
}

void scnn_net_backward(scnn_net *net, scnn_dtype *dy)
{
    if ((net == NULL) || (dy == NULL)) {
        return;
    }

    scnn_dtype *out = dy;
    for (int i = (net->size - 1); i >= 0; i--) {
        net->layers[i]->backward(net->layers[i], out);
        out = net->layers[i]->dx->data;
    }
}

void scnn_net_free(scnn_net **net)
{
    if (*net == NULL) {
        return;
    }

    for (int i = 0; i < (*net)->size; i++) {
        scnn_layer_free(&((*net)->layers[i]));
    }

    free(*net);
    *net = NULL;
}
