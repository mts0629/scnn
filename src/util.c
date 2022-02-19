/**
 * @file util.c
 * @brief utility functions and macros
 * 
 */
#include "util.h"

#include <stdlib.h>

void free_with_null(void **ptr)
{
    free(*ptr);
    *ptr = NULL;
}
