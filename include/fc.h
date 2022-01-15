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
 * @return Layer* pointer to layer
 */
Layer *fc_alloc(const LayerParameter layer_param);

#endif // FC_H
