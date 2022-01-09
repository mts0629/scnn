/**
 * @file mat.c
 * @brief matrix operations
 * 
 */
#include "mat.h"

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
float *mat_add(const float *a, const float *b, float *c, const int m, const int n)
{
    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }

    return c;
}

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
float *mat_mul(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            float y = 0;
            for (int k = 0; k < n; k++)
            {
                y += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

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
float *mat_mul_trans_a(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < p; j++)
        {
            float y = 0;
            for (int k = 0; k < m; k++)
            {
                y += a[k * n + i] * b[k * p + j];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

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
float *mat_mul_trans_b(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            float y = 0;
            for (int k = 0; k < n; k++)
            {
                y += a[i * n + k] * b[j * n + k];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

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
float *mat_mul_trans_ab(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            float y = 0;
            for (int k = 0; k < m; k++)
            {
                y += a[k * n + i] * b[j * m + k];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

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
float *mat_mul_scalar(const float *a, float *b, const int m, const int n, const float k)
{
    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        b[i] = k * a[i];
    }

    return b;
}
