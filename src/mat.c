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
