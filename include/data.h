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

/**
 * @brief randomize data array with uniform distribution [0, 1]
 * 
 * @param[out] array data array
 * @param[in] size num of elements
 */
void fdata_rand_uniform(float *array, const size_t size);

/**
 * @brief randomize data array with normal distribution
 * 
 * @param[out] array data array
 * @param[in] size num of elements
 * @param[in] mean mean of normal distribution
 * @param[in] std standard deviation of normal distribution
 */
void fdata_rand_norm(float *array, const size_t size, const float mean, const float std);

#endif // DATA_H
