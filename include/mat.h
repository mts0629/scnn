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
float *mat_alloc(const int, const int);

/**
 * @brief deallocate matrix
 * 
 */
void mat_free(float** mat);

/**
 * @brief copy matrix data
 * 
 * @return float* pointer to dest matrix
 */
float *mat_copy(const float*, const int, const int, float*);

/**
 * @brief add MxN matrix A and B: C=A+B
 * 
 * @return float* pointer to matrix C
 */
float *mat_add(const float*, const float*, float*, const int, const int);

/**
 * @brief multiply MxN matrix A and NxP matrix B: C=AB
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul(const float*, const float*, float*, const int, const int, const int);

/**
 * @brief multiply MxN matrix A and MxP matrix B: C=(A^T)B
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_a(const float*, const float*, float*, const int, const int, const int);

/**
 * @brief multiply MxN matrix A and PxM matrix B: C=A(B^T)
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_b(const float*, const float*, float*, const int, const int, const int);

/**
 * @brief multiply MxN matrix A and PxN matrix B: C=(A^T)B^T
 * 
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_ab(const float*, const float*, float*, const int, const int, const int);

/**
 * @brief mutiply MxN matrix A with scalar k: B=kA
 * 
 * @return float* pointer to matrix B
 */
float *mat_mul_scalar(const float*, float*, const int, const int, const float);

#endif // MAT_H
