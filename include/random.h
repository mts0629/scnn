/**
 * @file random.h
 * @brief pseudorandom number generators
 * 
 */
#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

/**
 * @brief set seed value of Xorshift PRNG
 * 
 * @param seed seed value
 */
void rand_seed(uint32_t seed);

/**
 * @brief get pseudorandom number by Xorshift PRNG
 * 
 * @return uint32_t pseudorandom number
 */
uint32_t rand_xorshift(void);

/**
 * @brief get pseudorandom number from uniform distribution [0, 1]
 * 
 * @return float pseudorandom number with [0, 1]
 */
float rand_uniform(void);

#endif // RANDOM_H
