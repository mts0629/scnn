/**
 * @file all_tests.c
 * @brief main module of test runner
 * 
 */
#include "unity_fixture.h"

static void RunAllTests(void)
{
    RUN_TEST_GROUP(mat);

    RUN_TEST_GROUP(layer);
}

int main(int argc, const char *argv[])
{
    return UnityMain(argc, argv, RunAllTests);
}
