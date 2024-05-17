//
// Created by hphi344 on 10/05/24.
//
#include <gtest/gtest.h>

#include "../include/gsDBSCAN.h"
#include <Eigen/Dense>
#include <chrono>
#include <arrayfire.h>

class gsDBSCANTest : public ::testing::Test {

};

TEST_F(gsDBSCANTest, TestFindDistances) {
    // Matrix X
    float dataX[] = {0, 1, 3,
                     1, 2, 0,
                     2, 0, 3,
                     3, 0, 1,
                     0, 0, 1};
    af::array X(5, 3, dataX);

    // Matrix A
    float dataA[] = {0, 3,
                     2, 5,
                     4, 1,
                     0, 7,
                     2, 1};
    af::array A(5, 2, dataA);

    // Matrix B
    float dataB[] = {1, 2, 3,
                     0, 4, 1,
                     3, 1, 0,
                     1, 0, 2,
                     0, 2, 3,
                     1, 2, 0,
                     0, 4, 1,
                     3, 1, 2,
                     1, 0, 4,
                     0, 2, 1};
    af::array B(10, 3, dataB);

    // Matrix initial
    float dataExpectedDistances[] = {11, 5, 14, 11, 0, 5,
                                     9, 0, 11, 0, 14, 11,
                                     5, 0, 5, 5, 8, 14,
                                     9, 5, 0, 0, 9, 5,
                                     9, 6, 5, 5, 0, 6};
    af::array expectedDistances(5, 6, dataExpectedDistances);

    auto start = std::chrono::high_resolution_clock::now();

    //distances = findDistances(X, A, B);

    // TODO

}

class TestCalculatingBatchSize : public gsDBSCANTest {

};

TEST_F(TestCalculatingBatchSize, TestLargeInput) {
    int batchSize = GsDBSCAN::findDistanceBatchSize(1, 1000000, 3, 2, 2000);

    ASSERT_EQ(20, batchSize);
}

TEST_F(TestCalculatingBatchSize, TestSmallInput) {
    int n = 100;

    int batchSize = GsDBSCAN::findDistanceBatchSize(1, n, 3, 10, 10);

    ASSERT_EQ(n, batchSize);
}

class TestFindingDistances : public gsDBSCANTest {

};

TEST_F(TestFindingDistances, TestSmallInput) {
    float X_data[] = {
            0, 1, 3,
            1, 2, 0,
            2, 0, 3,
            3, 0, 1,
            0, 0, 1
    };
    af::array X(5, 3, X_data);

    float A_data[] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };
    af::array A(5, 2, A_data);

    float B_data[] = {
            1, 2, 3,
            0, 4, 1,
            3, 1, 0,
            1, 0, 2,
            0, 2, 3,
            1, 2, 0,
            0, 4, 1,
            3, 1, 2,
            1, 0, 4,
            0, 2, 1
    };
    af::array B(10, 3, B_data);

    float expectedData[] = {
            11, 5, 14, 11, 0, 5,
            9, 0, 11, 0, 14, 11,
            5, 0, 5, 5, 8, 14,
            9, 5, 0, 0, 9, 5,
            9, 6, 5, 5, 0, 6
    };
    af::array expected = af::sqrt(af::array(5, 6, expectedData));

    af::array distances = GsDBSCAN::findDistances(X, A, B);

    // Check shape is (n, 2*k*m)
    ASSERT_TRUE(distances.dims(0) == X.dims(0) && distances.dims(1) == A.dims(1) * B.dims(1));

    ASSERT_TRUE(af::allTrue<bool>(af::abs(distances - expected) < 1e-6));
}