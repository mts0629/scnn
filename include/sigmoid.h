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
 * @return Layer* pointer to layer
 */
Layer *sigmoid_alloc(const LayerParameter layer_param);

#endif // SIGMOID_H
