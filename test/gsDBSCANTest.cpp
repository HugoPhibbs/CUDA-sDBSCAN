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
#include <matx.h>
namespace tu = testUtils;

class gsDBSCANTest : public ::testing::Test {

};

class TestArraySumThirdDimension : public gsDBSCANTest {

};

TEST_F(TestArraySumThirdDimension, TestSmallInput) {
    af::array array_in = af::randu(10, 5, 8, f32);

    af::array expected = af::sum(array_in, 2);

    af::array result = GsDBSCAN::arraySumThirdDim(array_in);

    af::print("", result);
    af::print("", expected);

    ASSERT_TRUE(tu::arraysApproxEqual(expected, result));
}

TEST_F(TestArraySumThirdDimension, TestLargeInput) {

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
            0, 1, 2, 3, 0,
            1, 2, 0, 0, 0,
            3, 0, 3, 1, 1
    };
    af::array X(5, 3, X_data);

    // k = 1
    int A_data[] = {
            0, 2, 4, 0, 2,
            3, 5, 1, 7, 1
    };
    af::array A(5, 2, A_data);

    // m = 3
    int B_data[] = {
            1, 0, 3, 1, 0, 1, 0, 3, 1, 0,
            2, 4, 1, 0, 2, 2, 4, 1, 0, 2,
            3, 1, 0, 2, 3, 0, 1, 2, 4, 1
    };
    af::array B(10, 3, B_data);

    float expectedData[] = {
            11, 9, 5, 9, 9,
            5, 0, 0, 5, 6,
            14, 11, 5, 0, 5,
            11, 0, 5, 0, 5,
            0, 14, 8, 9, 0,
            5, 11, 14, 5, 6
    };

    af::array expected = af::sqrt(af::array(5, 6, expectedData));


//    af::print("expected", expected);

    ASSERT_TRUE(expected.dims(0) == 5 && expected.dims(1) == 6); // Checking gtest is sane

//    af::array distances = findDistancesMatX(X, A, B);

    // Check shape is (n, 2*k*m)
    ASSERT_TRUE(distances.dims(0) == X.dims(0) && distances.dims(1) == A.dims(1) * B.dims(1));

    // May have something going on with column ordering
    af::print("distances", distances);
    af::print("expected", expected);
    af::array distancesSorted = af::sort(distances, 1);

    af::print("distancesSorted", distancesSorted);

    af::array expectedSortedData = af::sort(expected, 1);

    af::print("expectedSortedData", expectedSortedData);

//    af::print("distancesSorted", distancesSorted);
//    af::print("expectedSorted", expectedSortedData);

    ASSERT_TRUE(af::allTrue<bool>(af::abs(distancesSorted - expectedSortedData) < 1e-6));

}

TEST_F(TestFindingDistances, TestLargeInput) {
    af::array X = tu::createMockMnistDataset();
    af::array A, B;
    std::tie(A, B) = tu::createMockABMatrices();

    tu::Time start = tu::timeNow();

    af::array distances = GsDBSCAN::findDistances(X, A, B);

    tu::printDurationSinceStart(start);
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}