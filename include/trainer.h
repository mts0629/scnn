/**
 * @file trainer.h
 * @brief network training operations
 * 
 */
#ifndef TRAINER_H
#define TRAINER_H

#include "net.h"

/**
 * @brief train network with SGD (Stochastic Gradient Descent)
 * 
 * @param[in,out] net target network
 * @param[in] train_x array of training data
 * @param[in] train_t array of training labels
 * @param[in] test_x array of test data
 * @param[in] test_t array of test labels
 * @param[in] learning_rate learning rate
 * @param[in] epoch num of epochs
 * @param[in] train_data_size num of training data
 * @param[in] test_data_size num of test data
 * @param[in] loss_func loss function
 */
void train_sgd(
    Net *net,
    float **train_x,
    float **train_t,
    float **test_x,
    float **test_t,
    const float learning_rate,
    const int epoch,
    const int train_data_size,
    const int test_data_size,
    float (*loss_func)(const float*, const float*, const int));

#endif // TRAINER_H
