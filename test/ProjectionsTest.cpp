//
// Created by hphi344 on 9/08/24.
//
#include "../include/pch.h"
#include <gtest/gtest.h>
#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/TestUtils.h"
#include "../include/gsDBSCAN/run_utils.h"

namespace tu = testUtils;

class ProjectionsTest : public ::testing::Test {
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

    template <typename T>
    void assertArrayEqual(T* array1, T* array2, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(array1[i], array2[i], 1e-6);
        }
    }
};

class TestConstructingABMatrices : public ProjectionsTest {

};

TEST_F(TestConstructingABMatrices, TestSmallInputTorch) {
    // n = 6, D = 5
    float distances[30] = {
            12.0f, 85.0f, 47.0f, 23.0f, 56.0f, 10.0f,
            63.0f, 77.0f, 20.0f, 34.0f, 89.0f, 4.0f,
            45.0f, 90.0f, 27.0f, 69.0f, 10.0f, 3.0f,
            92.0f, 18.0f, 61.0f, 83.0f, 25.0f, 15.0f,
            51.0f, 39.0f, 74.0f, 6.0f, 81.0f, 2.0f
    }; // Remember, this is a column major array, so a 6x5 mat

    // A is (n, 2*k), or (6, 2*2) (row major)
    int expectedA[(6) * (2 * 2)] = {
            2 * 0, 2 * 2, 2 * 1 + 1, 2 * 3 + 1,
            2 * 3, 2 * 4, 2 * 0 + 1, 2 * 2 + 1,
            2 * 1, 2 * 2, 2 * 3 + 1, 2 * 4 + 1,
            2 * 4, 2 * 0, 2 * 2 + 1, 2 * 3 + 1,
            2 * 2, 2 * 3, 2 * 4 + 1, 2 * 1 + 1,
            2 * 4, 2 * 2, 2 * 0 + 1, 2 * 3 + 1
    };

    // B is (2*D, m), or (2*5, 2) (row major)
    int expectedB[(2 * 5) * 2] = {
            5, 0,
            4, 1,
            5, 2,
            1, 4,
            5, 4,
            3, 1,
            5, 1,
            3, 0,
            5, 3,
            2, 4
    };

    auto distances_d = GsDBSCAN::algo_utils::copyHostToDevice(distances, 30);
    auto distances_d_row_major = GsDBSCAN::algo_utils::colMajorToRowMajorMat(distances_d, 6, 5);

    auto distancesOptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto distances_tensor = torch::from_blob(distances_d_row_major, {6, 5}, distancesOptions);

    cudaCheckError();

    auto [A, B] = GsDBSCAN::projections::constructABMatricesTorch(distances_tensor, 2, 2);

    auto A_d = A.mutable_data_ptr<int>();
    auto B_d = B.mutable_data_ptr<int>();

    auto A_h = GsDBSCAN::algo_utils::copyDeviceToHost(A_d, 6 * 4);
    auto B_h = GsDBSCAN::algo_utils::copyDeviceToHost(B_d, 10 * 2);

    assertArrayEqual(expectedA, A_h, 6 * 4);
    assertArrayEqual(expectedB, B_h, 10 * 2);
}