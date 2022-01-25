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
 * @param[in] layer_param layer parameter
 * @return Layer* pointer to layer structure
 */
Layer *softmax_alloc(const LayerParameter layer_param);

#endif // SOFTMAX_H
