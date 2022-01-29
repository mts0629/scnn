/**
 * @file loss.c
 * @brief loss function
 * 
 */
#include "loss.h"

/**
 * @brief mean squared error (MSE)
 * 
 */
float mean_squared_error(const float *y, const float *t, const int n)
{
    float sq_err = 0;

    for (int i = 0; i < n; i++)
    {
        sq_err += (t[i] - y[i]) * (t[i] - y[i]);
    }

    return 0.5 * sq_err;
}
