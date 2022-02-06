/**
 * @file random.c
 * @brief pseudorandom number generators
 * 
 */
#include "random.h"

// initial seed values for XorShift with period 2^128 - 1
// these must be initialized to not be all zero
static uint32_t x = 123456789;
static uint32_t y = 362436069;
static uint32_t z = 521288629;
static uint32_t w = 88675123;

void rand_seed(uint32_t seed)
{
    // just replace one of the seeds by input
    x = 123456789;
    y = 362436069;
    z = 521288629;
    w = seed;
}

uint32_t rand_xorshift(void)
{
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    return w;
}

float rand_uniform(void)
{
    return (rand_xorshift() + 1.0f) / (UINT32_MAX + 2.0f);
}
