/**
 * @file loss.h
 * @brief loss function
 * 
 */
#ifndef LOSS_H
#define LOSS_H

/**
 * @brief mean squared loss (MSE)
 * 
 * @param y vector of predicted values 
 * @param t vector of target values 
 * @param size size of vector
 * @return float loss value
 */
float mean_squared_loss(const float *y, const float *t, const int size);

/**
 * @brief cross entropy loss
 * 
 * @param y vector of predicted values 
 * @param t vector of target values 
 * @param size size of vector
 * @return float loss value
 */
float cross_entropy_loss(const float *y, const float *t, const int size);

#endif // LOSS_H
