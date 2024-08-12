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


class TestEigenToMatXConversion : public UtilsTest {

};

TEST_F(TestEigenToMatXConversion, TestSmallInput) {
    Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, RowMajor> halfEigenMat(2, 2);

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

TEST_F(TestEigenToMatXConversion, TestABInput) {
    int n = 70000;
    int D = 1024;
    int k = 5;
    int m = 50;

    Matrix<int, -1, -1, RowMajor> AMat = (Eigen::MatrixXd::Random(n, 2*k) * (2 * (D - 1))).cast<int>();
    Matrix<int, -1, -1, RowMajor> BMat = (Eigen::MatrixXd::Random(2*D, m) * (n-1)).cast<int>();

    auto start = tu::timeNow();

    auto AMatXTensor = GsDBSCAN::eigenMatToMatXTensor<int, int>(AMat, matx::MATX_MANAGED_MEMORY);

    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start, "Time taken for A");

    start = tu::timeNow();

    auto BMatXTensor = GsDBSCAN::eigenMatToMatXTensor<int, int>(BMat, matx::MATX_MANAGED_MEMORY);

    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start, "Time taken for B");
}

TEST_F(TestEigenToMatXConversion, TestLargeInput) {
    int n = 100000;
    int d = 1000;

    Matrix<double, -1, -1, RowMajor> mat = Eigen::MatrixXd::Random(n, d);

    auto start = tu::timeNow();

    auto matXTensor = GsDBSCAN::eigenMatToMatXTensor<double, double>(mat, matx::MATX_MANAGED_MEMORY);

    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start, "Time taken to convert Eigen to MatX");

    // TODO fix the above! don't really want to use double here
}