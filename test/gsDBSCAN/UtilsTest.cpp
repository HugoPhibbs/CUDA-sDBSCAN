//
// Created by hphi344 on 9/08/24.
//
#include <gtest/gtest.h>
#include <matx.h>
#include <arrayfire.h>
#include <Eigen/Dense>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"

namespace tu = testUtils;

class UtilsTest : public ::testing::Test {

};

class TestAfToMatXConversion : public UtilsTest {

};

TEST_F(TestAfToMatXConversion, TestSmallInput) {
    float afData[] = {
            0, 1, 2, 3,
            0, 2, 1, 0,
            1, 8, 9, 11,
            15, 2, 6, 7
    };

    auto *afDataManaged = GsDBSCAN::hostToManagedArray<float>(afData, 16);

    af::array afArray(4, 4, afDataManaged);

    auto matXTensor = GsDBSCAN::afArrayToMatXTensor<float, float>(afArray);

    cudaDeviceSynchronize();

    float *matXData = matXTensor.Data();

    matx::print(matXTensor);
    //
    //    for (int i = 0; i < 16; i++) {
    //        ASSERT_NEAR(afDataManaged[i], matXData[i], 1e-6);
    //    }
}

class TestEigenToMatXConversion : public UtilsTest {

};

TEST_F(TestEigenToMatXConversion, TestSmallInput) {
    Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic> halfEigenMat(2, 2);

    halfEigenMat(0, 0) = Eigen::half(1.0);
    halfEigenMat(0, 1) = Eigen::half(2.0);
    halfEigenMat(1, 0) = Eigen::half(3.0);
    halfEigenMat(1, 1) = Eigen::half(4.0);

    auto matXTensor = GsDBSCAN::eigenMatToMatXTensor<Eigen::half, matx::matxFp16>(halfEigenMat, matx::MATX_MANAGED_MEMORY);

    cudaDeviceSynchronize();

    matx::matxFp16 *matXData = matXTensor.Data();
    Eigen::half *halfEigenData = halfEigenMat.data();

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(halfEigenData[i], matXData[i], 1e-6);
    }
}

TEST_F(TestEigenToMatXConversion, TestLargeInput) {
    int n = 100000;
    int d = 1000;

    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(n, d);

    auto start = tu::timeNow();

    auto matXTensor = GsDBSCAN::eigenMatToMatXTensor<double, double>(mat, matx::MATX_MANAGED_MEMORY);

    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start, "Time taken to convert Eigen to MatX");

    // TODO fix the above! don't really want to use double here
}