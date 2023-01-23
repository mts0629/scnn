/**
 * @file scnn_net.h
 * @brief Network structure
 * 
 */
#ifndef SCNN_NET_H
#define SCNN_NET_H

#include "scnn_layer.h"

#define SCNN_NET_MAX_SIZE 256  //!< The max number of layers in network

typedef struct scnn_net scnn_net;

/**
 * @brief Get the number of layers of a network
 * 
 * @param[in]   net Network
 * @return          The number of layers in the network
*/
int scnn_net_size(const scnn_net *net);

/**
 * @brief Get batch size of a network
 * 
 * @param[in]   net Network
 * @return          Batch size of the network
*/
int scnn_net_batch_size(const scnn_net *net);

/**
 * @brief Get layers in the network
 * 
 * @param[in]   net Network
 * @return          Pointer to layers in the network
*/
scnn_layer **scnn_net_layers(scnn_net *net);

/**
 * @brief Get an input layer of the network
 *
 * @param[in]   net Network
 * @return          Pointer to the input layer of network
*/
scnn_layer *scnn_net_input(const scnn_net *net);
 
/**
 * @brief Get an output layer of the network
 * @param[in]   net Network
 * @return          Pointer to the output layer of network
*/
scnn_layer *scnn_net_output(const scnn_net *net);

/**
 * @brief Allocate a network
 * 
 * @return  Pointer to the network, NULL if failed
 */
scnn_net *scnn_net_alloc(void);

/**
 * @brief Append a layer to the network
 * 
 * @param[in,out]   net     Network
 * @param[in]       layer   Layer
 * @return                  Pointer to the network, NULL if failed
 */
scnn_net *scnn_net_append(scnn_net *net, scnn_layer *layer);

/**
 * @brief Initialize a network
 * 
 * @param[in,out] net   Network
 * @return              Pointer to the network, NULL if failed
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
