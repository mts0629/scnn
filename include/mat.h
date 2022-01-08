#ifndef MAT_H
#define MAT_H

#include <stdbool.h>

float *mat_add(const float*, const float*, float*, const int, const int);

float *mat_mul(const float*, const float*, float*, const int, const int, const int);
float *mat_mul_trans_a(const float*, const float*, float*, const int, const int, const int);
float *mat_mul_trans_b(const float*, const float*, float*, const int, const int, const int);
float *mat_mul_trans_ab(const float*, const float*, float*, const int, const int, const int);

float *mat_mul_scalar(const float*, float*, const int, const int, const float);

#endif // MAT_H
