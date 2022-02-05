/**
 * @file test_random_runner.c
 * @brief test runner of random.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(random)
{
    RUN_TEST_CASE(random, get_rand_xorshift);
    RUN_TEST_CASE(random, get_rand_xorshift_with_same_seed);
    RUN_TEST_CASE(random, get_rand_xorshift_with_diff_seed);
}
