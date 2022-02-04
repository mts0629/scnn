/**
 * @file loss.c
 * @brief loss function
 * 
 */
#include "loss.h"

#include <math.h>

float mean_squared_error(const float *y, const float *t, const int n)
{
    float sq_err = 0;

    for (int i = 0; i < n; i++) {
        sq_err += (t[i] - y[i]) * (t[i] - y[i]);
    }

    return 0.5 * sq_err;
}

float cross_entropy_error(const float *y, const float *t, const int n)
{
    // small value to avoid log(0)
    const float epsilon = 1e-7;

    float err = 0;

    for (int i = 0; i < n; i++) {
        err += t[i] * log(y[i] + epsilon);
    }

    return -err;
}
