//
// Created by hphi344 on 9/08/24.
//
#include <gtest/gtest.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"
#include <cuda_runtime.h>

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
    float expectedB[(2 * 5) * 2] = {
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

    print("", A);

    auto A_array_d = A.device<float>();
    auto B_array_d = B.device<float>();

    auto A_array_h = GsDBSCAN::algo_utils::copyDeviceToHost(A_array_d, 6 * 4, GsDBSCAN::algo_utils::getAfCudaStream());
    auto B_array_h = GsDBSCAN::algo_utils::copyDeviceToHost(B_array_d, 10 * 2, GsDBSCAN::algo_utils::getAfCudaStream());

    assertColRowMajorMatsEqual(A_array_h, expectedA, 6, 4);
//    assertColRowMajorMatsEqual(B_array_h, expectedB, 10, 2);


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
    tu::printDurationSinceStart(start, "TestLargeInputArrayFire");
}

TEST_F(TestProjectionsSpeed, TestLargeInputMatx) {
    int n = 70000;
    int d = 784;
    int D = 1024;

    auto X = matx::random<float>({n, d}, matx::UNIFORM);
    auto Y = matx::random<float>({d, D}, matx::UNIFORM);

    auto start = tu::timeNow();
    auto Z = matx::make_tensor<float>({n, D});

    (Z = matx::matmul(X, Y)).run();
    cudaDeviceSynchronize(); // 300ms faster than ArrayFire, still, this makes preprocessing slower than CPU.
    /*
     * What is needed is a FHT on the GPU. Instead of a simple mat mul.
     */
    tu::printDurationSinceStart(start, "TestLargeInputMatx");
}

class TestPerformProjections : public ProjectionsTest {

};

//TEST_F(TestPerformProjections, TestSmallInputMatX) {
//    auto X = matx::random<float>({10, 3}, matx::UNIFORM);
//
//    auto start  = tu::timeNow();
//
//    auto projections = GsDBSCAN::projections::performProjectionsMatX(X, 3);
//
//    cudaDeviceSynchronize();
//
//    tu::printDurationSinceStart(start);
//}

//TEST_F(TestPerformProjections, TestLargeInputMatX) {
//    auto X = matx::random<float>({70000, 784}, matx::UNIFORM);
//
//    auto start = tu::timeNow();
//
//    auto projections = GsDBSCAN::projections::performProjectionsMatX(X, 50);
//
//    cudaDeviceSynchronize();
//
//    tu::printDurationSinceStart(start);
//}

TEST_F(TestPerformProjections, TestSmallInputAF) {

}

TEST_F(TestPerformProjections, TestLargeInputAF) {

}

class TestNormalisation : public ProjectionsTest {

};

TEST_F(TestNormalisation, TestSmallInputAF) {
    float X_data[] = {
            1.0f, 4.0f, 7.0f,
            2.0f, 5.0f, 8.0f,
            3.0f, 6.0f, 9.0f
    }; // Column major order

    auto X = af::array(3, 3, X_data);

    auto XNorm = GsDBSCAN::projections::normaliseDatasetAF(X);

    float expected[] = {
            1.0f / (float) std::sqrt(14), 4.0f / (float) std::sqrt(77), 7.0f / (float) std::sqrt(194),
            2.0f / (float) ::sqrt(14), 5.0f / (float) std::sqrt(77), 8.0f / (float) std::sqrt(194),
            3.0f / (float) std::sqrt(14), 6.0f / (float) std::sqrt(77), 9.0f / (float) std::sqrt(194)
    }; // Column major order


    auto XNorm_d = XNorm.device<float>();
    auto XNorm_h = GsDBSCAN::algo_utils::copyDeviceToHost(XNorm_d, 3 * 3, GsDBSCAN::algo_utils::getAfCudaStream());

    for (int i = 0; i < 3 * 3; ++i) {
        ASSERT_NEAR(expected[i], XNorm_h[i], 1e-6);
    }

    ASSERT_EQ(XNorm.dims(0), 3);
    ASSERT_EQ(XNorm.dims(1), 3);
}

TEST_F(TestNormalisation, TestLargeInputAF) {
    auto X = af::randu(70000, 784);

    auto start = tu::timeNow();

    auto XNorm = GsDBSCAN::projections::normaliseDatasetAF(X);
    XNorm.eval();

    tu::printDurationSinceStart(start);

    ASSERT_EQ(XNorm.dims(0), 70000);
    ASSERT_EQ(XNorm.dims(1), 784);
}

//TEST_F(TestNormalisation, TestSmallInputMatx) {
//    // TODO
//    float X_data[] = {
//            1.0f, 2.0f, 3.0f,
//            4.0f, 5.0f, 6.0f,
//            7.0f, 8.0f, 9.0f
//    }; // Row-major order
//
//    float expected[] = {
//            1.0f / (float) std::sqrt(14), 2.0f / (float) std::sqrt(14), 3.0f / (float) std::sqrt(14),
//            4.0f / (float) std::sqrt(77), 5.0f / (float) std::sqrt(77), 6.0f / (float) std::sqrt(77),
//            7.0f / (float) std::sqrt(194), 8.0f / (float) std::sqrt(194), 9.0f / (float) std::sqrt(194)
//    }; // Row-major order
//
//    auto *X_d = GsDBSCAN::algo_utils::copyHostToDevice(X_data, 3*3);
//    auto X = matx::make_tensor<float>(X_d, {3, 3});
////    auto XNorm = GsDBSCAN::projections::normaliseDatasetMatX(X);
//
//    auto XNorm_h = GsDBSCAN::algo_utils::copyDeviceToHost(XNorm.Data(), 3*3);
//
//    for (int i = 0; i < 3*3; ++i) {
//        ASSERT_NEAR(expected[i], XNorm_h[i], 1e-6);
//    }
//}

TEST_F(TestNormalisation, TestLargeInputMatx) {
    // TODO
}