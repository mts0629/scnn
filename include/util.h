/**
 * @file util.h
 * @brief utility functions and macros
 * 
 */
#ifndef UTIL_H
#define UTIL_H

/**
 * @brief free a memory block and set NULL
 * 
 * @param ptr address to pointer of memory block
 */
void free_with_null(void **ptr);

/**
 * @brief free a memory block and set NULL
 * @note need to specify address of pointer
 * 
 */
#define FREE_WITH_NULL(ptr) free_with_null((void**)(ptr))

#endif // UTIL_H
