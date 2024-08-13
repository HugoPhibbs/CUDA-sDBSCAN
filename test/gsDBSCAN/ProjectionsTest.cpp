//
// Created by hphi344 on 9/08/24.
//
#include <gtest/gtest.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"
#include <cuda_runtime.h>

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

TEST_F(TestProjectionsSpeed, TestLargeInputArrayFire) {
    int n = 70000;
    int d = 784;
    int D = 1024;

    auto X = af::randu(n, d);
    auto Y = af::randu(d, D);

    auto start = tu::timeNow();
    auto Z = af::matmul(X, Y); // Honestly a little too slow!
    af::eval(Z);
    cudaDeviceSynchronize(); // Honestly not sure if this is necessary here?
    tu::printDurationSinceStart(start);
}

TEST_F(TestProjectionsSpeed, TestLargeInputMatx) {
    int n = 70000;
    int d = 784;
    int D = 1024;

    auto X = matx::random<float>({n, d}, matx::UNIFORM);
    auto Y = matx::random<float>({d, D}, matx::UNIFORM);

    auto start = tu::timeNow();
    auto Z = matx::matmul(X, Y);
    Z.run();
    cudaDeviceSynchronize(); // 300ms faster than ArrayFire, still, this makes preprocessing slower than CPU.
    /*
     * What is needed is a FHT on the GPU. Instead of a simple mat mul.
     */
    tu::printDurationSinceStart(start);
}