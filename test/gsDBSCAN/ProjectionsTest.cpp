//
// Created by hphi344 on 9/08/24.
//
#include <gtest/gtest.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"

namespace tu = testUtils;

class ProjectionsTest: public ::testing::Test {
};

class TestConstructingABMatrices : public ProjectionsTest {

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

class TestProjectionsSpeed : public ProjectionsTest {

};

TEST_F(TestProjectionsSpeed, TestLargeInput) {
//    int n = 70000;
//    int d = 784;
//    int D = 1024;
//
//    auto X = matx::random<float>({n, d}, matx::UNIFORM);
//    auto Y = matx::random<float>({d, D}, matx::UNIFORM);
//    auto Z = matx::matmul(X, Y);
//
//    Z.run();
}