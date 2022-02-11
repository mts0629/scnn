/**
 * @file loss.h
 * @brief loss function
 * 
 */
#ifndef LOSS_H
#define LOSS_H

/**
 * @brief mean squared error (MSE)
 * 
 * @param y vector of predicted values 
 * @param t vector of target values 
 * @param size size of vector
 * @return float error
 */
float mean_squared_error(const float *y, const float *t, const int size);

/**
 * @brief cross entropy error
 * 
 * @param y vector of predicted values 
 * @param t vector of target values 
 * @param size size of vector
 * @return float error
 */
float cross_entropy_error(const float *y, const float *t, const int size);

#endif // LOSS_H
