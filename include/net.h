/**
 * @file net.h
 * @brief network structure
 * 
 */
#ifndef NET_H
#define NET_H

#include "layer.h"

/**
 * @struct
 * @brief network structure
 * 
 */
typedef struct Net {
    int length;     //!< num of layers
    Layer **layers; //!< list of layers

    Layer *input_layer;     //!< input layer
    Layer *output_layer;    //!< output layer
} Net;

/**
 * @brief create network
 * 
 * @param[in] length num of layers
 * @param[in] layers array of pointer of layer struct
 * @return Net* pointer to network structure
 */
Net *net_create(const int length, Layer *layers[]);

/**
 * @brief forward propagation of network
 * 
 * @param[in,out] net network structure
 * @param[in] x network input
 */
void net_forward(Net *net, const float *x);

/**
 * @brief backward propagation of network
 * 
 * @param[in,out] net network structure
 * @param[in] dy diff of network output
 */
void net_backward(Net *net, const float *dy);

/**
 * @brief deallocate network
 * 
 * @param[in,out] net network structure to be deallocated
 */
void net_free(Net **net);

#endif // NET_H
