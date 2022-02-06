/**
 * @file test_random.c
 * @brief unit test of random.c
 * 
 */
#include "random.h"

#include "unity_fixture.h"

TEST_GROUP(random);

TEST_SETUP(random)
{}

TEST_TEAR_DOWN(random)
{}

TEST(random, get_rand_xorshift)
{
    uint32_t rnd_s0[3];
    rnd_s0[0] = rand_xorshift();
    rnd_s0[1] = rand_xorshift();
    rnd_s0[2] = rand_xorshift();

    TEST_ASSERT_NOT_EQUAL_UINT32(3701687786, rnd_s0[1]);
    TEST_ASSERT_NOT_EQUAL_UINT32(458299110, rnd_s0[2]);
    TEST_ASSERT_NOT_EQUAL_UINT32(2500872618, rnd_s0[0]);
}

TEST(random, get_rand_xorshift_with_same_seed)
{
    rand_seed(0);

    uint32_t rnd_s0[3];
    rnd_s0[0] = rand_xorshift();
    rnd_s0[1] = rand_xorshift();
    rnd_s0[2] = rand_xorshift();

    rand_seed(0);

    uint32_t rnd_s1[3];
    rnd_s1[0] = rand_xorshift();
    rnd_s1[1] = rand_xorshift();
    rnd_s1[2] = rand_xorshift();

    TEST_ASSERT_EQUAL_UINT32(rnd_s1[0], rnd_s0[0]);
    TEST_ASSERT_EQUAL_UINT32(rnd_s1[1], rnd_s0[1]);
    TEST_ASSERT_EQUAL_UINT32(rnd_s1[2], rnd_s0[2]);
}

TEST(random, get_rand_xorshift_with_diff_seed)
{
    rand_seed(0);

    uint32_t rnd_s0[3];
    rnd_s0[0] = rand_xorshift();
    rnd_s0[1] = rand_xorshift();
    rnd_s0[2] = rand_xorshift();

    rand_seed(1);

    uint32_t rnd_s1[3];
    rnd_s1[0] = rand_xorshift();
    rnd_s1[1] = rand_xorshift();
    rnd_s1[2] = rand_xorshift();

    TEST_ASSERT_NOT_EQUAL_UINT32(rnd_s1[0], rnd_s0[0]);
    TEST_ASSERT_NOT_EQUAL_UINT32(rnd_s1[1], rnd_s0[1]);
    TEST_ASSERT_NOT_EQUAL_UINT32(rnd_s1[2], rnd_s0[2]);
}

TEST(random, get_rand_uniform)
{
    for (int i = 0; i < 10000; i++) {
        float rnd = rand_uniform();
        TEST_ASSERT_FLOAT_WITHIN(0.5, 0.5, rnd);
    }
}
