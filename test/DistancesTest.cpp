//
// Created by hphi344 on 9/08/24.
//

#include "../include/pch.h"
#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/TestUtils.h"
#include "../include/gsDBSCAN/run_utils.h"

namespace tu = testUtils;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                           \
        cudaError_t e = cudaGetLastError();                              \
        if (e != cudaSuccess) {                                          \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(e));                               \
            exit(EXIT_FAILURE);                                          \
        } else {                                                         \
            printf("CUDA call successful: %s:%d\n", __FILE__, __LINE__); \
        }                                                                \
    }

class TestDistances : public ::testing::Test {

protected:

    template <typename T>
    void assertArrayEqual(T* array1, T* array2, size_t size, T eps=1e-6) {
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(array1[i], array2[i], eps);
        }
    }
};

class TestCalculatingBatchSize : public TestDistances {

};

TEST_F(TestCalculatingBatchSize, TestLargeInput) {
    int batchSize = GsDBSCAN::distances::findDistanceBatchSize(1, 1000000, 3, 2, 2000);

    ASSERT_EQ(20, batchSize);
}

TEST_F(TestCalculatingBatchSize, TestSmallInput) {
    int n = 100;

    int batchSize = GsDBSCAN::distances::findDistanceBatchSize(1, n, 3, 10, 10);

    ASSERT_EQ(n, batchSize);
}


class TestFindingDistances : public TestDistances {

};

TEST_F(TestFindingDistances,TestSmallInputTorch) {
    float X[15] = {
            0, 1, 3,
            1, 2, 0,
            2, 0, 3,
            3, 0, 1,
            0, 0, 1
    };

    auto *X_d = GsDBSCAN::algo_utils::copyHostToDevice<float>(X, 15);

    int A[10] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };

    auto *A_d = GsDBSCAN::algo_utils::copyHostToDevice<int>(A, 10);

    int B[30] = {
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

    auto *B_d = GsDBSCAN::algo_utils::copyHostToDevice<int>(B, 30);

    auto X_torch = GsDBSCAN::algo_utils::torchTensorFromDeviceArray<float, torch::kFloat32>(X_d, 5, 3);
    auto A_torch = GsDBSCAN::algo_utils::torchTensorFromDeviceArray<int, torch::kInt32>(A_d, 5, 2);
    auto B_torch = GsDBSCAN::algo_utils::torchTensorFromDeviceArray<int, torch::kInt32>(B_d, 10, 3);

    auto distances = GsDBSCAN::distances::findDistancesTorch(X_torch, A_torch, B_torch);

    cudaDeviceSynchronize();

    auto distances_d = distances.mutable_data_ptr<float>();

    auto distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, 30);

    float expected_squared[30] = {
            11, 5, 14, 11, 0, 5,
            9, 0, 11, 0, 14, 11,
            5, 0, 5, 5, 8, 14,
            9, 5, 0, 0, 9, 5,
            9, 6, 5, 5, 0, 6
    };

    for (int i = 0; i < 5*6; i++) {
        ASSERT_NEAR(std::sqrt(expected_squared[i]), distances_h[i], 1e-3);
    }
}