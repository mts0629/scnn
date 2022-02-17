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
 * @param[in] x array of training data
 * @param[in] t array of training labels
 * @param[in] learning_rate learning rate
 * @param[in] epoch num of epochs
 * @param[in] data_size num of training data
 * @param[in] batch_size training batch size
 * @param[in] loss_func loss function
 */
void train_sgd(Net *net,
    float **x, float **t,
    const float learning_rate,
    const int epoch,
    const int data_size, const int batch_size,
    float (*loss_func)(const float*, const float*, const int));

#endif // TRAINER_H
