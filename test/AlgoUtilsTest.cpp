//
// Created by hphi344 on 9/08/24.
//

#include <omp.h>
#include "../include/pch.h"
#include <thrust/random.h>

#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/TestUtils.h"
#include "../include/gsDBSCAN/algo_utils.h"
#include "../include/gsDBSCAN/run_utils.h"

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
};

TEST_F(TestColToRowMajorArrayConversion, TestSmallInput) {
    const size_t numRows = 3;
    const size_t numCols = 4;

    float h_colMajorArray[numRows * numCols] = {
            1.0f, 4.0f, 7.0f, 10.0f,
            2.0f, 5.0f, 8.0f, 11.0f,
            3.0f, 6.0f, 9.0f, 12.0f
    };

    float *d_colMajorArray = GsDBSCAN::algo_utils::copyHostToDevice(h_colMajorArray, numRows * numCols);

    float *d_rowMajorArray = GsDBSCAN::algo_utils::colMajorToRowMajorMat(d_colMajorArray, numRows, numCols);

    float *h_rowMajorArray = GsDBSCAN::algo_utils::copyDeviceToHost(d_rowMajorArray, numRows * numCols);

    assertColRowMajorMatsEqual(h_colMajorArray, h_rowMajorArray, numRows, numCols);

    cudaFree(d_colMajorArray);
    cudaFree(d_rowMajorArray);
    delete[] h_rowMajorArray;
}

class TestRowToColMajorArrayConversion : public AlgoUtilsTest {
};

TEST_F(TestRowToColMajorArrayConversion, TestSmallInput) {
    const size_t numRows = 3;
    const size_t numCols = 4;

    float h_rowMajorArray[numRows * numCols] = {
            1.0f, 4.0f, 7.0f, 10.0f,
            2.0f, 5.0f, 8.0f, 11.0f,
            3.0f, 6.0f, 9.0f, 12.0f
    };

    float *d_rowMajorArray = GsDBSCAN::algo_utils::copyHostToDevice(h_rowMajorArray, numRows * numCols);

    float *d_colMajorArray = GsDBSCAN::algo_utils::rowMajorToColMajorMat(d_rowMajorArray, numRows, numCols);

    float* h_colMajorArray = GsDBSCAN::algo_utils::copyDeviceToHost(d_colMajorArray, numRows * numCols);

    assertColRowMajorMatsEqual(h_colMajorArray, h_rowMajorArray, numRows, numCols);

    cudaFree(d_colMajorArray);
    cudaFree(d_rowMajorArray);
    delete[] h_colMajorArray;
}

class TestCopying : public AlgoUtilsTest {

};

struct random_functor {
    __host__ __device__
    float operator()(unsigned int thread_id) {
        thrust::default_random_engine rng(thread_id);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }
};

TEST_F(TestCopying, TestLargeInputDeviceToHost) {
    const int N = 100000*1000;

    // Create a thrust device vector with 100000 elements
    thrust::device_vector<float> d_vec(N);

    // Fill the vector with random numbers
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_vec.begin(),
                      random_functor());

    auto d_array = thrust::raw_pointer_cast(d_vec.data());

    auto start = tu::timeNow();

    auto h_array = GsDBSCAN::algo_utils::copyDeviceToHost(d_array, N);

    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start);
}

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