/**
 * @file test_trainer_runner.c
 * @brief test runner of trainer.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(trainer)
{
    RUN_TEST_CASE(trainer, train_sgd);
}
