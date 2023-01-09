/**
 * @file scnn_net.h
 * @brief Network structure
 * 
 */
#ifndef SCNN_NET_H
#define SCNN_NET_H

#include "scnn_layer.h"

#define SCNN_NET_MAX_SIZE 256  //!< The max number of layers in network

/**
 * @brief Network structure
 * 
 */
typedef struct scnn_net {
    int         size;                       //!< The number of layers
    int         batch_size;                 //!< Batch size
    scnn_layer  *layers[SCNN_NET_MAX_SIZE]; //!< Layers
    scnn_layer  *input;                     //!< Input layer off the network
    scnn_layer  *output;                    //!< Output layer off the network
} scnn_net;

/**
 * @brief Allocate network
 * 
 * @return  Pointer to network, NULL if failed
 */
scnn_net *scnn_net_alloc(void);

/**
 * @brief Append layer to network
 * 
 * @param[in,out]   net     Network
 * @param[in]       params  Parameters of the layer to be appended
 * @return                  Pointer to network, NULL if failed
 */
scnn_net *scnn_net_append(scnn_net *net, scnn_layer_params params);

/**
 * @brief Initialize network
 * 
 * @param[in,out] net   Network
 * @return              Pointer to network, NULL if failed
 */
scnn_net *scnn_net_init(scnn_net *net);

/**
 * @brief Forward propagation of network
 * 
 * @param[in,out]   net Network
 * @param[in]       x   Network input
 */
void scnn_net_forward(scnn_net *net, scnn_dtype *x);

/**
 * @brief Backward propagation of network
 * 
 * @param[in,out]   net Network
 * @param[in]       dy  Differential of network output
 */
void scnn_net_backward(scnn_net *net, scnn_dtype *dy);

/**
 * @brief Free network
 * 
 * @param[in,out] net   Pointer to poiner of network
 */
void scnn_net_free(scnn_net **net);

#endif // SCNN_NET_H
