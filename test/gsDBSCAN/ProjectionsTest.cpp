//
// Created by hphi344 on 9/08/24.
//
#include <gtest/gtest.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"
#include <cuda_runtime.h>

namespace tu = testUtils;

class ProjectionsTest: public ::testing::Test {
    protected:
    template<typename T>
    void assertColRowMajorMatsEqual(T *colMajorArray, T *rowMajorArray, size_t numRows, size_t numCols) {

        #pragma omp parallel for
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                T colMajorValue = colMajorArray[j * numRows + i];
                T rowMajorValue = rowMajorArray[i * numCols + j];
                printf("%zu, %zu\n", i, j);
                ASSERT_NEAR(colMajorValue, rowMajorValue, 1e-6);
            }
        }
    }
};

class TestConstructingABMatrices : public ProjectionsTest {

};

TEST_F(TestConstructingABMatrices, TestMediumInputAF) {
    int n = 100;
    int D = 30;
    int k = 5;
    int m = 10;
    std::string filename = "./data/projections.csv";
    af::array projections = tu::csvToArray(filename);

    int a = 1;

    af::array A, B;

    std::tie(A, B) = GsDBSCAN::projections::constructABMatricesAF(projections, k, m);

    ASSERT_TRUE(A.dims(0) == n && A.dims(1) == k);
    ASSERT_TRUE(B.dims(0) == n && B.dims(1) == m);
}

TEST_F(TestConstructingABMatrices, TestSortSpeedAF) {
    int n = 70000;
    int D = 784;
    int k = 5;
    int m = 50;

    auto distances = tu::createMockDistances(n, D);

    distances.eval();

    auto start = tu::timeNow();

    auto sorted = af::sort(distances, 0); // Down the cols
    sorted.eval();

    tu::printDurationSinceStart(start); // Very fast. 0.018 seconds

    auto start2 = tu::timeNow();

    auto sorted2 = af::sort(distances, 1); // Across the rows
    sorted2.eval();

    tu::printDurationSinceStart(start2); // Very fast. 0.010 seconds
}

TEST_F(TestConstructingABMatrices, TestSmallInputAF) {

    // n = 6, D = 5
    float distances[30] = {
            12.0f, 85.0f, 47.0f, 23.0f, 56.0f, 10.0f,
            63.0f, 77.0f, 20.0f, 34.0f, 89.0f, 4.0f,
            45.0f, 90.0f, 27.0f, 69.0f, 10.0f, 3.0f,
            92.0f, 18.0f, 61.0f, 83.0f, 25.0f, 15.0f,
            51.0f, 39.0f, 74.0f, 6.0f, 81.0f, 2.0f
    }; // Remember, this is a column major array, so a 6x5 mat

    af::array distancesAF(6, 5, distances);

    af::array indices0, indices1;
    af::array temp0, temp1;
    af::sort(temp0, indices0, distancesAF, 0);
    af::sort(temp1, indices1, distancesAF, 1);

    af::print("", indices0);
    af::print("", indices1);

    // Take k=m=2

    // A is (n, 2*k), or (6, 2*2) (row major)
    float expectedA[(6) * (2 * 2)] = {
            2 * 0, 2 * 2, 2 * 1 + 1, 2 * 3 + 1,
            2 * 3, 2 * 4, 2 * 0 + 1, 2 * 2 + 1,
            2 * 1, 2 * 2, 2 * 3 + 1, 2 * 4 + 1,
            2 * 4, 2 * 0, 2 * 2 + 1, 2 * 3 + 1,
            2 * 2, 2 * 3, 2 * 4 + 1, 2 * 1 + 1,
            2 * 4, 2 * 2, 2 * 0 + 1, 2 * 3 + 1
    };

    // B is (2*D, m), or (2*5, 2) (row major)
    float expectedB[(2*5) * 2] = {
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

    // Yes I did the above manually

    auto [A, B] = GsDBSCAN::projections::constructABMatricesAF(distancesAF, 2, 2);

    auto A_array_d = A.device<float>();
    auto B_array_d = B.device<float>();

    auto A_array_h = GsDBSCAN::utils::copyDeviceToHost(A_array_d, 6*4, GsDBSCAN::utils::getAfCudaStream());
    auto B_array_h = GsDBSCAN::utils::copyDeviceToHost(B_array_d, 10*2, GsDBSCAN::utils::getAfCudaStream());

    assertColRowMajorMatsEqual(expectedA, A_array_h, 6, 4);
//    assertColRowMajorMatsEqual(expectedB, B_array_h, 10, 2);


    // TODO, do this with CuPy to cross reference the results
}

TEST_F(TestConstructingABMatrices, TestLargeInputAF) {

    int n = 70000;
    int D = 1024;
    int k = 5;
    int m = 50;

    auto distances = tu::createMockDistances(n, D);

    auto start = tu::timeNow();

    // TODO figure out how to use sort_index

    auto [A, B] = GsDBSCAN::projections::constructABMatricesAF(distances, k, m);
    A.eval();
    B.eval();

    tu::printDurationSinceStart(start);
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