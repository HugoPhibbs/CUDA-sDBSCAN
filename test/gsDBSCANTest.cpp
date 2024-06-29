//
// Created by hphi344 on 10/05/24.
//
#include <gtest/gtest.h>

#include "../include/GsDBSCAN.h"
#include <Eigen/Dense>
#include <chrono>
#include <arrayfire.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

/**
 * Reads a CSV file and returns a vector of vectors of floats
 *
 * Used for testing, can use python as a baseline to generate test data
 *
 * @param filename
 * @return
 * @return
 */
std::vector<std::vector<float>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stof(cell));
        }

        data.push_back(row);
    }

    return data;
}

af::array csvToArray(const std::string& filename) {
    std::vector<std::vector<float>> data = readCSV(filename);
    int n = data.size();
    int m = data.at(0).size();

    af::array array(n, m, f32); // Create array with matching dimensions and data type (f32 for float)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array(i, j) = data[i][j];
        }
    }

    return array;
}

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
    af::array projections = csvToArray(filename);

    int a = 1;

    af::array A, B;

    std::tie(A, B) = GsDBSCAN::constructABMatrices(projections, k, m);

    ASSERT_TRUE(A.dims(0) == n && A.dims(1) == k);
    ASSERT_TRUE(B.dims(0) == n && B.dims(1) == m);
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

    ASSERT_TRUE(expected.dims(0) == 5 && expected.dims(1) == 6); // Checking gtest is sane


//    af::array distances = GsDBSCAN::findDistances(X, A, B);
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
        0, 2, 1, 0,
        0, 2, 1, 0
    };

    af::array distances(4, 4, distancesData);

    float eps = 2.1;

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);
    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    distances.eval();
    E.eval();
    V.eval();

    // TODO
}