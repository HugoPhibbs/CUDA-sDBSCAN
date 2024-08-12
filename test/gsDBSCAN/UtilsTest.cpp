//
// Created by hphi344 on 9/08/24.
//
#include <gtest/gtest.h>
#include <matx.h>
#include <arrayfire.h>
#include <Eigen/Dense>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"
#include <omp.h>

namespace tu = testUtils;

class UtilsTest : public ::testing::Test {

protected:
    template<typename T>
    void assertColRowMajorMatsEqual(T *colMajorArray, T *rowMajorArray, size_t numRows, size_t numCols) {

        #pragma omp parallel for
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                T colMajorValue = colMajorArray[j * numRows + i];
                T rowMajorValue = rowMajorArray[i * numCols + j];
                ASSERT_NEAR(colMajorValue, rowMajorValue, 1e-6);
            }
        }
    }
};

class TestColToRowMajorArrayConversion : public UtilsTest {

protected:
    template <typename T>
    T* afArrayToHostArray(af::array afArray) {
        T* deviceArray = afArray.device<float>();
        return GsDBSCAN::copyDeviceToHost(deviceArray, afArray.elements(), GsDBSCAN::getAfCudaStream());
    }
};

TEST_F(TestColToRowMajorArrayConversion, TestSmallInput) {
    const size_t numRows = 3;
    const size_t numCols = 4;

    float h_colMajorArray[numRows * numCols] = {
            1.0f, 4.0f, 7.0f, 10.0f,
            2.0f, 5.0f, 8.0f, 11.0f,
            3.0f, 6.0f, 9.0f, 12.0f
    };

    float *d_colMajorArray;
    cudaMalloc(&d_colMajorArray, numRows * numCols * sizeof(float));
    cudaMemcpy(d_colMajorArray, h_colMajorArray, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);

    float *d_rowMajorArray = GsDBSCAN::colMajorToRowMajorMat(d_colMajorArray, numRows, numCols);

    float h_rowMajorArray[numRows * numCols];
    cudaMemcpy(h_rowMajorArray, d_rowMajorArray, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    assertColRowMajorMatsEqual(h_colMajorArray, h_rowMajorArray, numRows, numCols);

    cudaFree(d_colMajorArray);
    cudaFree(d_rowMajorArray);
}

TEST_F(TestColToRowMajorArrayConversion, TestLargeInput) {
    const int n = 70000;
    const int d = 1024;

    auto afArray = af::randu(n, d, f32);
    afArray.eval();

    auto afCudaStream = GsDBSCAN::getAfCudaStream();

    float *colMajorMat_d = afArray.device<float>();

    // Running the test

    auto start = tu::timeNow();

    float *rowMajorMat_d = GsDBSCAN::colMajorToRowMajorMat(colMajorMat_d, n, d, afCudaStream);

    tu::printDurationSinceStart(start);

    // Now copy back to the host and compare the two arrays

    auto rowMajorMat_h = GsDBSCAN::copyDeviceToHost(rowMajorMat_d, n * d, afCudaStream);

    auto colMajorMat_h = GsDBSCAN::copyDeviceToHost(colMajorMat_d, n * d, afCudaStream);

    assertColRowMajorMatsEqual(colMajorMat_h, rowMajorMat_h, n, d);

    afArray.unlock();

    cudaFree(rowMajorMat_d);
    free(colMajorMat_h);
    free(rowMajorMat_h);
}

class TestArrayFireToMatXConversion : public UtilsTest {

};

TEST_F(TestArrayFireToMatXConversion, TestSmallInput) {
    const size_t numRows = 3;
    const size_t numCols = 4;

    float h_colMajorArray[numRows * numCols] = {
            1.0f, 4.0f, 7.0f, 10.0f,
            2.0f, 5.0f, 8.0f, 11.0f,
            3.0f, 6.0f, 9.0f, 12.0f
    };

    af::array afArray(numRows, numCols, h_colMajorArray);

    auto matXTensor = GsDBSCAN::afArrayToMatXTensor<float, float>(afArray);

    float *matxTensor_d = matXTensor.Data();
    float *afArray_d = afArray.device<float>();

    auto *matxTensor_h = GsDBSCAN::copyDeviceToHost(matxTensor_d, numRows*numCols, GsDBSCAN::getAfCudaStream());
    float *afArray_h = GsDBSCAN::copyDeviceToHost(afArray_d, numRows*numCols, GsDBSCAN::getAfCudaStream());

    assertColRowMajorMatsEqual(afArray_h, matxTensor_h, numRows, numCols);

    free(matxTensor_h);
    free(afArray_h);
    afArray.unlock();
}

TEST_F(TestArrayFireToMatXConversion, TestLargeInput) {
    const int n = 70000;
    const int d = 1024;

    auto afArray = af::randu(n, d, f32);

    auto start = tu::timeNow();

    auto matXTensor = GsDBSCAN::afArrayToMatXTensor<float, float>(afArray);

    tu::printDurationSinceStart(start);

    float *matxTensor_d = matXTensor.Data();
    float *afArray_d = afArray.device<float>();

    auto *matxTensor_h = GsDBSCAN::copyDeviceToHost(matxTensor_d, n*d, GsDBSCAN::getAfCudaStream());
    float *afArray_h = GsDBSCAN::copyDeviceToHost(afArray_d, n*d, GsDBSCAN::getAfCudaStream());

    assertColRowMajorMatsEqual(afArray_h, matxTensor_h, n, d);

    free(matxTensor_h);
    free(afArray_h);
    afArray.unlock();
}


class TestEigenToMatXConversion : public UtilsTest {

};

TEST_F(TestEigenToMatXConversion, TestSmallInput) {
    Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, RowMajor> halfEigenMat(2, 2);

    halfEigenMat(0, 0) = Eigen::half(1.0);
    halfEigenMat(0, 1) = Eigen::half(2.0);
    halfEigenMat(1, 0) = Eigen::half(3.0);
    halfEigenMat(1, 1) = Eigen::half(4.0);

    auto matXTensor = GsDBSCAN::eigenMatToMatXTensor<Eigen::half, matx::matxFp16>(halfEigenMat,
                                                                                  matx::MATX_MANAGED_MEMORY);
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

    Matrix<int, -1, -1, RowMajor> AMat = (Eigen::MatrixXd::Random(n, 2 * k) * (2 * (D - 1))).cast<int>();
    Matrix<int, -1, -1, RowMajor> BMat = (Eigen::MatrixXd::Random(2 * D, m) * (n - 1)).cast<int>();

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