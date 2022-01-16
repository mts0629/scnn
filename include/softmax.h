/**
 * @file softmax.h
 * @brief Softmax layer
 * 
 */
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"

/**
 * @brief allocate Softmax layer
 * 
 * @return Layer* pointer to layer
 */
Layer *softmax_alloc(const LayerParameter layer_param);

#endif // SOFTMAX_H
