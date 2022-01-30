/**
 * @file mat.c
 * @brief matrix operations
 * 
 */
#include "mat.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief allocate MxN matrix
 * 
 * @param[in] m num of rows
 * @param[in] n  num of columns
 * @return float* pointer to matrix
 */
float *mat_alloc(const int m, const int n)
{
    if ((m < 1) || (n < 1))
    {
        return NULL;
    }

    float *mat = malloc(sizeof(float) * m * n);
    if (mat == NULL)
    {
        return NULL;
    }

    return mat;
}

/**
 * @brief deallocate matrix
 * 
 * @param[out] mat address of pointer to matrix
 */
void mat_free(float **mat)
{
    free(*mat);
    *mat = NULL;
}

/**
 * @brief fill matrix with specified value
 * 
 * @param mat target matrix
 * @param m num of rows
 * @param n  num of columns
 * @param value filling value
 * @return float* pointer to matrix
 */
float *mat_fill(float *mat, const int m, const int n, const float value)
{
    if (mat == NULL)
    {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        mat[i] = value;
    }

    return mat;
}

/**
 * @brief allocate MxN matrix filled with 0
 * 
 * @param[in] m num of rows
 * @param[in] n  num of columns
 * @return float* pointer to matrix
 */
float *mat_zeros(const int m, const int n)
{
    float *mat = mat_alloc(m, n);
    if (mat == NULL)
    {
        return NULL;
    }

    return mat_fill(mat, m, n, 0);
}

/**
 * @brief copy matrix data
 * 
 * @param src src MxN matrix
 * @param m matrix rows
 * @param n matrix columns
 * @param dst dest MxN matrix
 * @return float* pointer to dest matrix
 */
float *mat_copy(const float *src, const int m, const int n, float *dest)
{
    if ((src == NULL) || (dest == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1))
    {
        return NULL;
    }

    memcpy(dest, src, (sizeof(float) * m * n));

    return dest;
}

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
    if ((a == NULL) || (b == NULL) | (c == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1))
    {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }

    return c;
}

/**
 * @brief subtract MxN matrix A and B: C=A+B
 * 
 */
float *mat_sub(const float *a, const float *b, float *c, const int m, const int n)
{
    if ((a == NULL) || (b == NULL) | (c == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1))
    {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] - b[i];
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
    if ((a == NULL) || (b == NULL) | (c == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1))
    {
        return NULL;
    }

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
    if ((a == NULL) || (b == NULL) | (c == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1))
    {
        return NULL;
    }

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
    if ((a == NULL) || (b == NULL) | (c == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1))
    {
        return NULL;
    }

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
    if ((a == NULL) || (b == NULL) | (c == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1))
    {
        return NULL;
    }

    for (int i = 0; i < n; i++)
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
    if ((a == NULL) || (b == NULL))
    {
        return NULL;
    }

    if ((m < 1) || (n < 1))
    {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        b[i] = k * a[i];
    }

    return b;
}
