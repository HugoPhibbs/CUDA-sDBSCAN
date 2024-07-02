//
// Created by hphi344 on 10/05/24.
//
#include <gtest/gtest.h>

#include "../include/GsDBSCAN.h"
#include "../include/TestUtils.h"
#include <Eigen/Dense>
#include <chrono>
#include <arrayfire.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
namespace tu = testUtils;

class gsDBSCANTest : public ::testing::Test {

};

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

class TestConstructingABMatrices : public gsDBSCANTest {

};

TEST_F(TestConstructingABMatrices, TestSmallInput) {
    int n = 100;
    int D = 30;
    int k = 5;
    int m = 10;
    std::string filename = "./data/projections.csv";
    af::array projections = tu::csvToArray(filename);

    int a = 1;

    af::array A, B;

    std::tie(A, B) = GsDBSCAN::constructABMatrices(projections, k, m);

    ASSERT_TRUE(A.dims(0) == n && A.dims(1) == k);
    ASSERT_TRUE(B.dims(0) == n && B.dims(1) == m);
}

class TestFindingDistances : public gsDBSCANTest {

};

TEST_F(TestFindingDistances, TestSmallInput)     {
    // n = 5, d = 3
    float X_data[] = {
            0, 1, 3,
            1, 2, 0,
            2, 0, 3,
            3, 0, 1,
            0, 0, 1
    };
    af::array X(5, 3, X_data);

    // k = 1
    float A_data[] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };
    af::array A(5, 2, A_data);

    // m = 3
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

    ASSERT_TRUE(expected.dims(0) == 5 && expected.dims(1) == 6); // Checking gtest is sane


    af::array distances = GsDBSCAN::findDistances(X, A, B);
//
//    // Check shape is (n, 2*k*m)
//    ASSERT_TRUE(distances.dims(0) == X.dims(0) && distances.dims(1) == A.dims(1) * B.dims(1));
//
//    ASSERT_TRUE(af::allTrue<bool>(af::abs(distances - expected) < 1e-6));
}

class TestConstructQueryVectorDegreeArray : public gsDBSCANTest {

};

TEST_F(TestConstructQueryVectorDegreeArray, TestSmallInput) {
    float distancesData[] = {
            0, 1, 2, 3,
            0, 2, 1, 0,
            1, 8, 9, 11,
            15, 2, 6, 7
    };

    af::array distances(4, 4, distancesData);

    float eps = 2.1;

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);

    float expectedData[] = {3, 4, 1, 1};

    af::array expected(1, 4, expectedData);

    ASSERT_TRUE(af::allTrue<bool>(expected == E));
}

TEST_F(TestConstructQueryVectorDegreeArray, TestMnist) {
    af::array distances = tu::createMockDistances();

    float eps = af::randu(1, 1).scalar<float>();

    tu::Time start = tu::timeNow();

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);

    E.eval();
    af::sync();

    tu::printDurationSinceStart(start);
}

class TestProcessQueryVectorDegreeArray : public gsDBSCANTest {

};

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInput) {

    float EData[] = {3, 4, 1, 1};
    af::array E(1, 4, EData);

    float VExpectedData[] = {0, 3, 7, 8};

    af::array VExpected(1, 4, VExpectedData);

    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    ASSERT_TRUE(af::allTrue<bool>(VExpected ==  V));
}

TEST_F(TestProcessQueryVectorDegreeArray, TestMnist) {
    af::array distances = tu::createMockDistances();

    float eps = af::randu(1, 1).scalar<float>();

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);

    tu::Time start = tu::timeNow();

    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    V.eval();
    af::sync();

    tu::printDurationSinceStart(start);
}