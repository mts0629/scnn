/**
 * @file test_loss.c
 * @brief unit test of loss.c
 * 
 */
#include "loss.h"

#include "unity_fixture.h"

TEST_GROUP(loss);

TEST_SETUP(loss)
{}

TEST_TEAR_DOWN(loss)
{}

TEST(loss, mean_squared_loss)
{
    float x[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    float t[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

    float loss = mean_squared_loss(x, t, 10);

    TEST_ASSERT_EQUAL_FLOAT(0.0975, loss);
}

TEST(loss, cross_entropy_loss)
{
    float x[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    float t[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

    float loss = cross_entropy_loss(x, t, 10);

    TEST_ASSERT_EQUAL(6.447238, loss);
}
