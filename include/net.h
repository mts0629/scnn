/**
 * @file net.h
 * @brief network structure
 * 
 */
#ifndef NET_H
#define NET_H

#include "layer.h"

#define NET_NAME_MAX_LENGTH 32  //!< max length of network name

/**
 * @struct
 * @brief network structure
 * 
 */
typedef struct
{
    char name[NET_NAME_MAX_LENGTH + 1]; //!< network name

    int num_layers; //!< num of layers
    Layer **layers; //!< layer list
} Net;

/**
 * @brief create network
 * 
 * @param[in] name name of network
 * @param[in] length num of layers
 * @param[in] layers list of layers
 * @return Net* pointer to network structure
 */
Net *net_create(const char *name, const int length, Layer **layers);

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
