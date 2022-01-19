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

    Layer **layers; //!< layer list
} Net;

/**
 * @brief create network
 * 
 * @param name network name
 * @param length num of layers
 * @param layers layer list
 * @return Net* pointer to network
 */
Net *net_create(const char *name, const int length, Layer **layers);

/**
 * @brief deallocate network
 * 
 * @param net network to deallocate
 */
void net_free(Net **net);

#endif // NET_H
