/**
 * @file data.c
 * @brief data structures and operations
 * 
 */
#include "data.h"

#include <stdlib.h>
#include <string.h>

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
