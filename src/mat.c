/**
 * @file mat.c
 * @brief matrix operations
 * 
 */
#include "mat.h"

#include <stddef.h>

float *mat_add(const float *a, const float *b, float *c, const int m, const int n)
{
    if ((a == NULL) || (b == NULL) | (c == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1)) {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }

    return c;
}

float *mat_sub(const float *a, const float *b, float *c, const int m, const int n)
{
    if ((a == NULL) || (b == NULL) | (c == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1)) {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++) {
        c[i] = a[i] - b[i];
    }

    return c;
}

float *mat_mul(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    if ((a == NULL) || (b == NULL) | (c == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1)) {
        return NULL;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float y = 0;
            for (int k = 0; k < n; k++) {
                y += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

float *mat_mul_trans_a(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    if ((a == NULL) || (b == NULL) | (c == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1)) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float y = 0;
            for (int k = 0; k < m; k++) {
                y += a[k * n + i] * b[k * p + j];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

float *mat_mul_trans_b(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    if ((a == NULL) || (b == NULL) | (c == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1)) {
        return NULL;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float y = 0;
            for (int k = 0; k < n; k++) {
                y += a[i * n + k] * b[j * n + k];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

float *mat_mul_trans_ab(const float *a, const float *b, float *c, const int m, const int n, const int p)
{
    if ((a == NULL) || (b == NULL) | (c == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1) || (p < 1)) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float y = 0;
            for (int k = 0; k < m; k++) {
                y += a[k * n + i] * b[j * m + k];
            }
            c[i * p + j] = y;
        }
    }

    return c;
}

float *mat_mul_scalar(const float *a, float *b, const int m, const int n, const float k)
{
    if ((a == NULL) || (b == NULL)) {
        return NULL;
    }

    if ((m < 1) || (n < 1)) {
        return NULL;
    }

    const int size = m * n;
    for (int i = 0; i < size; i++) {
        b[i] = k * a[i];
    }

    return b;
}
