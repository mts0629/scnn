/**
 * @file net.h
 * @brief network structure
 * 
 */
#ifndef NET_H
#define NET_H

#include "layer.h"

#define NET_LAYER_MAX_SIZE 256  //!< max size of layers

/**
 * @struct
 * @brief network structure
 * 
 */
typedef struct Net {
    int   size;                        //!< num of layers
    Layer *layers[NET_LAYER_MAX_SIZE]; //!< list of layers
    Layer *input_layer;     //!< input layer
    Layer *output_layer;    //!< output layer
} Net;

/**
 * @brief allocate network
 * 
 * @return Net* pointer to network structute
 */
Net *net_alloc(void);

/**
 * @brief create network
 * 
 * @param[in] size num of layers
 * @param[in] layers array of pointer of layer struct
 * @return Net* pointer to network structure
 */
Net *net_create(const int size, Layer *layers[]);

/**
 * @brief append layer to network
 * 
 * @param[in,out] net target network
 * @param[in] layer layer to be appended
 * @return Net* pointer to the network structure
 */
Net *net_append(Net *net, Layer *layer);

/**
 * @brief initialize layer parameters in network
 * 
 * @param[in,out] net target network
 */
void net_init_layer_params(Net *net);

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
 * @param[in] t training label
 */
void net_backward(Net *net, const float *t);

/**
 * @brief deallocate network
 * 
 * @param[in,out] net network structure to be deallocated
 */
void net_free(Net **net);

#endif // NET_H
