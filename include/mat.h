/**
 * @file mat.h
 * @brief matrix operations
 * 
 */
#ifndef MAT_H
#define MAT_H

#include <stdlib.h>
#include <stdbool.h>

/**
 * @brief allocate MxN matrix
 * 
 * @return float* pointer to matrix
 */
float *mat_alloc(const int m, const int n);

/**
 * @brief deallocate matrix
 * 
 */
void mat_free(float **mat);

/**
 * @brief copy matrix data
 * 
 * @return float* pointer to dest matrix
 */
float *mat_copy(const float *src, const int m, const int n, float* dest);

/**
 * @brief add MxN matrix A and B: C=A+B
 * 
 * @return float* pointer to matrix C
 */
float *mat_add(const float *a, const float *b, float *c, const int m, const int n);

/**
 * @brief multiply MxN matrix A and NxP matrix B: C=AB
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief multiply MxN matrix A and MxP matrix B: C=(A^T)B
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_a(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief multiply MxN matrix A and PxM matrix B: C=A(B^T)
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_b(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief multiply MxN matrix A and PxN matrix B: C=(A^T)B^T
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_ab(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief mutiply MxN matrix A with scalar k: B=kA
 * 
 * @return float* pointer to matrix B
 */
float *mat_mul_scalar(const float *a, float *b, const int m, const int n, const float k);

#endif // MAT_H
