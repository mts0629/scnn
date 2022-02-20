/**
 * @file data.c
 * @brief data structures and operations
 * 
 */
#include "data.h"

#include <stdlib.h>
#include <string.h>

#include "random.h"

float* fdata_alloc(const size_t size)
{
    return (float*)malloc(sizeof(float) * size);
}

void fdata_copy(const float *src, const size_t size, float *dest)
{
    if ((src == NULL) || (dest == NULL)) {
        return;
    }
    memcpy(dest, src, (sizeof(float) * size));
}

void fdata_rand_uniform(float *array, const size_t size)
{
    if ((array == NULL) || (size < 1)) {
        return;
    }

    for (size_t i = 0; i < size; i++) {
        array[i] = rand_uniform();
    }
}

void fdata_rand_norm(float *array, const size_t size, const float mean, const float std)
{
    if ((array == NULL) || (size < 1) || (std < 0)) {
        return;
    }

    for (size_t i = 0; i < size; i++) {
        array[i] = rand_norm(mean, std);
    }
}
