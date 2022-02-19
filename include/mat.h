/**
 * @file mat.h
 * @brief matrix operations
 * 
 */
#ifndef MAT_H
#define MAT_H

#include <stdbool.h>

/**
 * @brief randomize matrix with uniform distribution [0, 1]
 * 
 * @param[out] mat target matrix
 * @param[in] size size of matrix
 * @return float* pointer to matrix
 */
float *mat_randomize_uniform(float *mat, const int size);

/**
 * @brief randomize matrix with normal distribution
 * 
 * @param[out] mat target matrix
 * @param[in] size size of matrix
 * @param[in] mean mean of normal distribution
 * @param[in] std standard deviation of normal distribution
 * @return float* pointer to matrix
 */
float *mat_randomize_norm(float *mat, const int size, const float mean, const float std);

/**
 * @brief add MxN matrix A and B: C=A+B
 * 
 * @param[in] a MxN matrix A
 * @param[in] b MxN matrix B
 * @param[out] c MxN matrix C
 * @param[in] m num of rows of matrix A/B
 * @param[in] n num of columns of matrix A/B
 * @return float* pointer to matrix C
 */
float *mat_add(const float *a, const float *b, float *c, const int m, const int n);

/**
 * @brief subtract MxN matrix A and B: C=A-B
 * 
 * @param[in] a MxN matrix A
 * @param[in] b MxN matrix B
 * @param[out] c MxN matrix C
 * @param[in] m num of rows of matrix A/B
 * @param[in] n num of columns of matrix A/B
 * @return float* pointer to matrix C
 */
float *mat_sub(const float *a, const float *b, float *c, const int m, const int n);

/**
 * @brief multiply MxN matrix A and NxP matrix B: C=AB
 * 
 * @param[in] a MxN matrix A
 * @param[in] b NxP matrix B
 * @param[out] c MxP matrix C
 * @param[in] m num of rows of matrix A
 * @param[in] n num of columns of matrix A/rows of matrix B
 * @param[in] p num of columns of matrix B
 * @return float* pointer to matrix C
 */
float *mat_mul(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief multiply MxN matrix A and MxP matrix B: C=(A^T)B
 * 
 * @param[in] a MxN matrix A
 * @param[in] b MxP matrix B
 * @param[out] c NxP matrix C
 * @param[in] m num of rows of matrix A/rows of matrix B
 * @param[in] n num of columns of matrix A
 * @param[in] p num of columns of matrix B
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_a(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief multiply MxN matrix A and PxN matrix B: C=A(B^T)
 * 
 * @param[in] a MxN matrix A
 * @param[in] b PxN matrix B
 * @param[out] c MxP matrix C
 * @param[in] m num of rows of matrix A
 * @param[in] n num of columns of matrix A/columns of matrix B
 * @param[in] p num of rows of matrix B
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_b(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief multiply MxN matrix A and PxN matrix B: C=(A^T)B^T
 * 
 * @param[in] a MxN matrix A
 * @param[in] b PxM matrix B
 * @param[out] c NxP matrix C
 * @param[in] m num of rows of matrix A
 * @param[in] n num of columns of matrix A/columns of matrix B
 * @param[in] p num of rows of matrix B
 * @return float* pointer to matrix C
 */
float *mat_mul_trans_ab(const float *a, const float *b, float *c, const int m, const int n, const int p);

/**
 * @brief mutiply MxN matrix A with scalar k: B=kA
 * 
 * @param[in] a MxN matrix A
 * @param[out] b MxN matrix B
 * @param[in] m num of rows of matrix A/B
 * @param[in] n num of columns of matrix A/B
 * @param[in] k coeffcient k
 * @return float* pointer to matrix B
 */
float *mat_mul_scalar(const float *a, float *b, const int m, const int n, const float k);

#endif // MAT_H
