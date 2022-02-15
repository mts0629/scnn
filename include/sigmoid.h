/**
 * @file sigmoid.h
 * @brief Sigmoid layer
 * 
 */
#ifndef SIGMOID_H
#define SIGMOID_H

#include "layer.h"

/**
 * @brief allocate Sigmoid layer
 * 
 * @param[in] layer_param layer parameter
 * @return Layer* pointer to layer structure
 */
Layer *sigmoid_layer(const LayerParameter layer_param);

#endif // SIGMOID_H
