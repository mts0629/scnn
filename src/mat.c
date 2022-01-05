#include "mat.h"

float *mat_add(const float *a, const float *b, float *c, const int m, const int n)
{
    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }

    return c;
}

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

float *mat_mul_scalar(const float *a, float *b, const int m, const int n, const float k)
{
    const int size = m * n;
    for (int i = 0; i < size; i++)
    {
        b[i] = k * a[i];
    }

    return b;
}
