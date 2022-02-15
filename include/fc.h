/**
 * @file fc.h
 * @brief Fully connected layer
 * 
 */
#ifndef FC_H
#define FC_H

#include "layer.h"

/**
 * @brief allocate Fully connected layer
 * 
 * @param[in] layer_param layer parameter
 * @return Layer* poiner to layer structure
 */
Layer *fc_layer(const LayerParameter layer_param);

#endif // FC_H
