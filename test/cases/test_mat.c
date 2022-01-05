#include "mat.h"

#include "unity_fixture.h"

TEST_GROUP(mat);

TEST_SETUP(mat)
{}

TEST_TEAR_DOWN(mat)
{}

TEST(mat, test_mat_add)
{
    float a[3 * 2] = {
        0, 1,
        2, 3,
        4, 5 
    };

    float b[3 * 2] = {
        -2, -1,
        0, 1,
        2, 3 
    };

    float c[3 * 2];

    float y[3 * 2] = {
        -2, 0,
        2, 4,
        6, 8 
    };

    mat_add(a, b, c, 3, 2);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, c, (3 * 2));
}
