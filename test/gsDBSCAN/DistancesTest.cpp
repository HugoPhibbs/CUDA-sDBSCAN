//
// Created by hphi344 on 9/08/24.
//

#include <gtest/gtest.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"

namespace tu = testUtils;

class TestDistances : public ::testing::Test {

};

class TestCalculatingBatchSize : public TestDistances {

};

TEST_F(TestCalculatingBatchSize, TestLargeInput) {
    int batchSize = GsDBSCAN::findDistanceBatchSize(1, 1000000, 3, 2, 2000);

    ASSERT_EQ(20, batchSize);
}

TEST_F(TestCalculatingBatchSize, TestSmallInput) {
    int n = 100;

    int batchSize = GsDBSCAN::findDistanceBatchSize(1, n, 3, 10, 10);

    ASSERT_EQ(n, batchSize);
}


class TestFindingDistances : public TestDistances {

};

TEST_F(TestFindingDistances, TestSmallInput)     {
    // n = 5, d = 3
    float X_data[] = {
            0, 1, 2, 3, 0,
            1, 2, 0, 0, 0,
            3, 0, 3, 1, 1
    };
    af::array X(5, 3, X_data);

    // k = 1
    int A_data[] = {
            0, 2, 4, 0, 2,
            3, 5, 1, 7, 1
    };
    af::array A(5, 2, A_data);

    // m = 3
    int B_data[] = {
            1, 0, 3, 1, 0, 1, 0, 3, 1, 0,
            2, 4, 1, 0, 2, 2, 4, 1, 0, 2,
            3, 1, 0, 2, 3, 0, 1, 2, 4, 1
    };
    af::array B(10, 3, B_data);

    float expectedData[] = {
            11, 9, 5, 9, 9,
            5, 0, 0, 5, 6,
            14, 11, 5, 0, 5,
            11, 0, 5, 0, 5,
            0, 14, 8, 9, 0,
            5, 11, 14, 5, 6
    };

    af::array expected = af::sqrt(af::array(5, 6, expectedData));


//    af::print("expected", expected);

    ASSERT_TRUE(expected.dims(0) == 5 && expected.dims(1) == 6); // Checking gtest is sane

    af::array distances = GsDBSCAN::findDistances(X, A, B);

    af::print("distances", af::pow(distances, 2));

    // Check shape is (n, 2*k*m)
    ASSERT_TRUE(distances.dims(0) == X.dims(0) && distances.dims(1) == A.dims(1) * B.dims(1));

    // May have something going on with column ordering
    af::print("distances", distances);
    af::print("expected", expected);
    af::array distancesSorted = af::sort(distances, 1);

    af::print("distancesSorted", distancesSorted);

    af::array expectedSortedData = af::sort(expected, 1);

    af::print("expectedSortedData", expectedSortedData);

//    af::print("distancesSorted", distancesSorted);
//    af::print("expectedSorted", expectedSortedData);

    ASSERT_TRUE(af::allTrue<bool>(af::abs(distancesSorted - expectedSortedData) < 1e-6));

}

template <typename T>
T* hostArrayToCudaArray(const T* hostArray, size_t numElements) {
    T* deviceArray;
    size_t size = sizeof(T) * numElements;
    cudaMalloc(&deviceArray, size);
    cudaMemcpy(deviceArray, hostArray, size, cudaMemcpyHostToDevice);
    return deviceArray;
}

TEST_F(TestFindingDistances, TestSmallInputMatx) {
    float X[15] = {
            0, 1, 3,
            1, 2, 0,
            2, 0, 3,
            3, 0, 1,
            0, 0, 1
    };

    float *X_d = hostArrayToCudaArray<float>(X, 15);

    int A[10] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };

    int *A_d = hostArrayToCudaArray<int>(A, 10);

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

    int *B_d = hostArrayToCudaArray<int>(B, 30);

    auto X_t = matx::make_tensor<float>(X_d, {5, 3}, true);
    matx::tensor_t<matx::matxFp16, 2> X_t_16({5, 3});
    (X_t_16 = matx::as_type<matx::matxFp16>(X_t)).run();
    auto A_t = matx::make_tensor<int>(A_d, {5, 2}, true);
    auto B_t = matx::make_tensor<int>(B_d, {10, 3}, true);

    auto distances_t = GsDBSCAN::findDistancesMatX(X_t_16, A_t, B_t);

    cudaDeviceSynchronize();

    cudaCheckError();

    matx::matxFp16 *distances_ptr = distances_t.Data();

    matx::matxFp16 expected_squared[] = {
            11, 5, 14, 11, 0, 5,
            9, 0, 11, 0, 14, 11,
            5, 0, 5, 5, 8, 14,
            9, 5, 0, 0, 9, 5,
            9, 6, 5, 5, 0, 6
    };

    for (int i = 0; i < 5*6; i++) {
        ASSERT_NEAR(std::sqrt(expected_squared[i]), distances_ptr[i], 1e-3);
    }
}

TEST_F(TestFindingDistances, TestSmallInputBatchingMatx) {
    float X[15] = {
            0, 1, 3,
            1, 2, 0,
            2, 0, 3,
            3, 0, 1,
            0, 0, 1
    };

    float *X_d = hostArrayToCudaArray<float>(X, 15);

    int A[10] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };

    int *A_d = hostArrayToCudaArray<int>(A, 10);

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

    int *B_d = hostArrayToCudaArray<int>(B, 30);

    auto X_t = matx::make_tensor<float>(X_d, {5, 3}, true);
    matx::tensor_t<matx::matxFp16, 2> X_t_16({5, 3});
    (X_t_16 = matx::as_type<matx::matxFp16>(X_t)).run();
    auto A_t = matx::make_tensor<int>(A_d, {5, 2}, true);
    auto B_t = matx::make_tensor<int>(B_d, {10, 3}, true);

    auto distances_t = GsDBSCAN::findDistancesMatX(X_t_16, A_t, B_t, 1.2, 1);

    cudaDeviceSynchronize();

    matx::matxFp16 *distances_ptr = distances_t.Data();

    matx::matxFp16 expected_squared[] = {
            11, 5, 14, 11, 0, 5,
            9, 0, 11, 0, 14, 11,
            5, 0, 5, 5, 8, 14,
            9, 5, 0, 0, 9, 5,
            9, 6, 5, 5, 0, 6
    };

    for (int i = 0; i < 5*6; i++) {
        ASSERT_NEAR(std::sqrt(expected_squared[i]), distances_ptr[i], 1e-3);
    }
}

template <typename T>
std::vector<T> loadCsvColumnToVector(const std::string& filePath, size_t columnIndex = 1) {
    rapidcsv::Document csvDoc(filePath);
    return csvDoc.GetColumn<T>(columnIndex);
}


TEST_F(TestFindingDistances, TestMediumInputMatx) {
    /*
     * This test checks if results calculated by C++/MatX are identical to those with Python/CuPy
     */

    auto AVector = loadCsvColumnToVector<int>("/home/hphi344/Documents/Thesis/python/data/A_n1000_k3.csv");
    int *A_h = AVector.data();

    auto BVector = loadCsvColumnToVector<int>("/home/hphi344/Documents/Thesis/python/data/B_D100_m20.csv");
    int *B_h = BVector.data();

    auto XVector = loadCsvColumnToVector<float>("/home/hphi344/Documents/Thesis/python/data/X_n1000_d20.csv");
    float *X_h = XVector.data();

    auto distancesVector = loadCsvColumnToVector<float>("/home/hphi344/Documents/Thesis/python/data/distances_n1000_k3_m20.csv");

    float *distances_expected_h = distancesVector.data();

    int n = 1000;
    int k = 3;
    int m = 20;
    int D = 100;
    int d = 20;

    int *A_d = hostArrayToCudaArray(A_h, n*2*k);
    int *B_d = hostArrayToCudaArray(B_h, 2*D*m);
    float *X_d = hostArrayToCudaArray(X_h, n*d);

    auto X_t = matx::make_tensor<float>(X_d, {n, d});
    matx::tensor_t<matx::matxFp16, 2> X_t_16({n, d});
    (X_t_16 = matx::as_type<matx::matxFp16>(X_t)).run();

    auto A_t = matx::make_tensor<int>(A_d, {n, 2 * k});
    auto B_t = matx::make_tensor<int>(B_d, {2 * D, m});

    auto start = tu::timeNow();

    auto distances_t = GsDBSCAN::findDistancesMatX(X_t_16, A_t, B_t, 1.2, 100);

    cudaDeviceSynchronize();

    matx::matxFp16 *distances_ptr = distances_t.Data();

    for (int i = 0; i < n*2*k*m; i++) {
        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
        ASSERT_NEAR(distances_expected_h[i], matx::promote_half_t<matx::matxFp16>(distances_ptr[i]), 1e-2); // Doing a cast just to be sure
    }
}



TEST_F(TestFindingDistances, TestLargeInputMatX) {
    int k = 5;
    int n = 70000;
    int m = 50;
    int D = 1024;
    int d = 784;

    auto A = tu::createMockAMatrixMatX(n, k, D);
    auto B = tu::createMockBMatrixMatX(n, m, D);
    auto X = tu::createMockMnistDatasetMatX(n, d);

    cudaDeviceSynchronize();

    tu::Time start = tu::timeNow();

    auto distances = GsDBSCAN::findDistancesMatX(X, A, B, 1.2, 250);
    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start); // This is too fn slow. Around 14 seconds, Cupy takes less than 0.7 seconds.

    printf("%lld %lld", distances.Shape()[0], distances.Shape()[1]);

    ASSERT_TRUE(distances.Shape()[0] == n);
    ASSERT_TRUE(distances.Shape()[1] == 2*k*m);
}

TEST_F(TestFindingDistances, TestLargeInput) {
    int n = 70000;
    af::array X = tu::createMockMnistDataset(n);
    af::array A, B;
    std::tie(A, B) = tu::createMockABMatrices(n);

    cudaDeviceSynchronize();

    tu::Time start = tu::timeNow();

    af::array distances = GsDBSCAN::findDistances(X, A, B);

    tu::printDurationSinceStart(start);
}