/**
 * @file data.h
 * @brief data structures and operations
 * 
 */
#ifndef DATA_H
#define DATA_H

#include <stddef.h>

/**
 * @brief allocate float data array
 * 
 * @param[in] size num of elements
 * @return float* pointer to array
 */
float* fdata_alloc(const size_t size);

/**
 * @brief copy float data from src to dest
 * 
 * @param[in] src source array
 * @param[in] size num of elements
 * @param[out] dest destination array
 */
void fdata_copy(const float *src, const size_t size, float *dest);

#endif // DATA_H
