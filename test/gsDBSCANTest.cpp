//
// Created by hphi344 on 10/05/24.
//
#include <gtest/gtest.h>

#include "../src/dbscan/gsDBSCAN.cpp"
#include <Eigen/Dense>
#include <chrono>
#include <arrayfire.h>

class gsDBSCANTest : public ::testing::Test {

};

TEST_F(gsDBSCANTest, TestFindDistances) {
    using namespace Eigen;
    Eigen::MatrixXf X(5, 3);
    X << 0, 1, 3,
            1, 2, 0,
            2, 0, 3,
            3, 0, 1,
            0, 0, 1;

    Eigen::MatrixXf A(5, 2);
    A << 0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1;

    Eigen::MatrixXf B(10, 3);
    B << 1, 2, 3,
            0, 4, 1,
            3, 1, 0,
            1, 0, 2,
            0, 2, 3,
            1, 2, 0,
            0, 4, 1,
            3, 1, 2,
            1, 0, 4,
            0, 2, 1;


    Eigen::MatrixXf initial(5, 6);
    initial << 11, 5, 14, 11, 0, 5,
            9, 0, 11, 0, 14, 11,
            5, 0, 5, 5, 8, 14,
            9, 5, 0, 0, 9, 5,
            9, 6, 5, 5, 0, 6;

    auto start = std::chrono::high_resolution_clock::now();




}