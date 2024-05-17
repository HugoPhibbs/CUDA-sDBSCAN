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
    af::array X(1000000, 3);
    af::array D(1000, 1000);

    float alpha = 1;

    GsDBSCAN gsDbscan(X, D, 10, 2, 2000, 0.1, true);
    int batchSize = gsDbscan.findDistanceBatchSize(alpha);

    ASSERT_EQ(20, batchSize);
}

TEST_F(TestCalculatingBatchSize, TestSmallInput) {
    af::array X(100, 3);
    af::array D(10, 10);

    float alpha = 1;

    GsDBSCAN gsDbscan(X, D, 10, 10, 10, 0.1, true);
    int batchSize = gsDbscan.findDistanceBatchSize(alpha);

    ASSERT_EQ(X.dims(0), batchSize);
}