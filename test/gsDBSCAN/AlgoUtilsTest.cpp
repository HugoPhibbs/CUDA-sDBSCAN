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

class AlgoUtilsTest : public ::testing::Test {

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

class TestColToRowMajorArrayConversion : public AlgoUtilsTest {

protected:
    template <typename T>
    T* afArrayToHostArray(af::array afArray) {
        T* deviceArray = afArray.device<float>();
        return GsDBSCAN::algo_utils::copyDeviceToHost(deviceArray, afArray.elements(), GsDBSCAN::algo_utils::getAfCudaStream());
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

    float *d_rowMajorArray = GsDBSCAN::algo_utils::colMajorToRowMajorMat(d_colMajorArray, numRows, numCols);

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

    auto afCudaStream = GsDBSCAN::algo_utils::getAfCudaStream();

    float *colMajorMat_d = afArray.device<float>();

    // Running the test

    auto start = tu::timeNow();

    float *rowMajorMat_d = GsDBSCAN::algo_utils::colMajorToRowMajorMat(colMajorMat_d, n, d, afCudaStream);

    tu::printDurationSinceStart(start);

    // Now copy back to the host and compare the two arrays

    auto rowMajorMat_h = GsDBSCAN::algo_utils::copyDeviceToHost(rowMajorMat_d, n * d, afCudaStream);

    auto colMajorMat_h = GsDBSCAN::algo_utils::copyDeviceToHost(colMajorMat_d, n * d, afCudaStream);

    assertColRowMajorMatsEqual(colMajorMat_h, rowMajorMat_h, n, d);

    afArray.unlock();

    cudaFree(rowMajorMat_d);
    free(colMajorMat_h);
    free(rowMajorMat_h);
}

class TestArrayFireToMatXConversion : public AlgoUtilsTest {

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

    auto matXTensor = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(afArray);

    float *matxTensor_d = matXTensor.Data();
    float *afArray_d = afArray.device<float>();

    auto *matxTensor_h = GsDBSCAN::algo_utils::copyDeviceToHost(matxTensor_d, numRows * numCols, GsDBSCAN::algo_utils::getAfCudaStream());
    float *afArray_h = GsDBSCAN::algo_utils::copyDeviceToHost(afArray_d, numRows * numCols, GsDBSCAN::algo_utils::getAfCudaStream());

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

    auto matXTensor = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(afArray);

    tu::printDurationSinceStart(start);

    float *matxTensor_d = matXTensor.Data();
    float *afArray_d = afArray.device<float>();

    auto *matxTensor_h = GsDBSCAN::algo_utils::copyDeviceToHost(matxTensor_d, n * d, GsDBSCAN::algo_utils::getAfCudaStream());
    float *afArray_h = GsDBSCAN::algo_utils::copyDeviceToHost(afArray_d, n * d, GsDBSCAN::algo_utils::getAfCudaStream());

    assertColRowMajorMatsEqual(afArray_h, matxTensor_h, n, d);

    free(matxTensor_h);
    free(afArray_h);
    afArray.unlock();
}


class TestCopying : public AlgoUtilsTest {

};

TEST_F(TestCopying, TestSmallInput) {
    float h_array[5] = {1, 2, 3, 4, 5};

    float *d_array = GsDBSCAN::algo_utils::copyHostToDevice(h_array, 5);

    float *h_array_copy = GsDBSCAN::algo_utils::copyDeviceToHost(d_array, 5);

    for (int i = 0; i < 5; ++i) {
        ASSERT_NEAR(h_array[i], h_array_copy[i], 1e-6);
    }
}

TEST_F(TestCopying, TestSmallInputMatx) {
    float arr[5] = {1, 2, 3, 4, 5};
    float *arr_d = GsDBSCAN::algo_utils::copyHostToDevice(arr, 5);

    auto tensor = matx::make_tensor<float>(arr_d, {5}, matx::MATX_DEVICE_MEMORY);

    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, tensor.Data());

    float *arr_copy = GsDBSCAN::algo_utils::copyDeviceToHost(tensor.Data(), 5);

    std::cout << "Pointer type: " << attributes.type << std::endl;

    for (int i = 0; i < 5; ++i) {
        ASSERT_NEAR(arr[i], arr_copy[i], 1e-6);
    }
}