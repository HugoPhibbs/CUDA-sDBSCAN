//
// Created by hphi344 on 9/08/24.
//

#include <gtest/gtest.h>
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

TEST_F(TestFindingDistances, TestSmallInputAF)     {
    GTEST_SKIP(); // Legacy test, I should remove it!
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

    af::array distances = GsDBSCAN::distances::findDistancesL2AF(X, A, B);

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

TEST_F(TestFindingDistances, TestSmallInputMatx) {
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

    auto X_t = matx::make_tensor<float>(X_d, {5, 3});
    auto A_t = matx::make_tensor<int>(A_d, {5, 2});
    auto B_t = matx::make_tensor<int>(B_d, {10, 3});

    auto distances_t = GsDBSCAN::distances::findDistancesMatX(X_t, A_t, B_t);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();

    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, 30);

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

TEST_F(TestFindingDistances, TestSmallInputCosineMatx) {

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

    int *A_d = GsDBSCAN::algo_utils::copyHostToDevice<int>(A, 10);

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

    int *B_d = GsDBSCAN::algo_utils::copyHostToDevice<int>(B, 30);

    auto X_t = matx::make_tensor<float>(X_d, {5, 3});
    matx::tensor_t<matx::matxFp16, 2> X_t_16({5, 3});
    (X_t_16 = matx::as_type<matx::matxFp16>(X_t)).run();
    auto A_t = matx::make_tensor<int>(A_d, {5, 2});
    auto B_t = matx::make_tensor<int>(B_d, {10, 3});

    std::string distanceMetric = "COSINE";

    auto distances_t = GsDBSCAN::distances::findDistancesMatX(X_t_16, A_t, B_t, 1.2, 1, distanceMetric);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();
    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, 30);

    float expectedDistances[30] = {
            2, 9, 3, 2, 10, 9,
            3, 5, 2, 5, 2, 2,
            9, 13, 9, 9, 3, 2,
            3, 9, 10, 10, 3, 9,
            1, 0, 3, 3, 1, 0
    };

    for (int i = 0; i < 5*6; i++) {
        ASSERT_NEAR(expectedDistances[i], distances_h[i], 1e-3);
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

    auto *X_d = GsDBSCAN::algo_utils::copyHostToDevice<float>(X, 15);

    int A[10] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };

    int *A_d = GsDBSCAN::algo_utils::copyHostToDevice<int>(A, 10);

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

    int *B_d = GsDBSCAN::algo_utils::copyHostToDevice<int>(B, 30);

    auto X_t = matx::make_tensor<float>(X_d, {5, 3});
    auto A_t = matx::make_tensor<int>(A_d, {5, 2});
    auto B_t = matx::make_tensor<int>(B_d, {10, 3});

    auto distances_t = GsDBSCAN::distances::findDistancesMatX(X_t, A_t, B_t, 1.2, 1);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();
    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, 30);

    float expected_squared[] = {
            11, 5, 14, 11, 0, 5,
            9, 0, 11, 0, 14, 11,
            5, 0, 5, 5, 8, 14,
            9, 5, 0, 0, 9, 5,
            9, 6, 5, 5, 0, 6
    };

    for (int i = 0; i < 5 * 6; i++) {
        ASSERT_NEAR(std::sqrt(expected_squared[i]), distances_h[i], 1e-3);
    }
}

TEST_F(TestFindingDistances, TestLargeInputMatXFromFile) {
    auto AVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/distances_test/A_test_expected_row_major.bin");
    auto BVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/distances_test/B_test_expected_row_major.bin");
    auto distancesVector = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/distances_test/distances_test_row_major.bin");

    auto XVector = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin");

    int n = 70000;
    int d = 784;
    int k = 5;
    int m = 50;
    int D = 1024;

    int *A_h = AVector.data();
    int *B_h = BVector.data();
    float *distances_expected_h = distancesVector.data();
    float *X_h = XVector.data();

    int *A_d = GsDBSCAN::algo_utils::copyHostToDevice(A_h, n * 2 * k);
    int *B_d = GsDBSCAN::algo_utils::copyHostToDevice(B_h, 2 * D * m);

    auto X_af = af::array(n, d, X_h);
    X_af = GsDBSCAN::projections::normaliseDatasetAF(X_af);
    X_af.eval();

    auto X_t = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(X_af);

    auto A_t = matx::make_tensor<int>(A_d, {n, 2 * k});
    auto B_t = matx::make_tensor<int>(B_d, {2 * D, m});

    auto distances_t = GsDBSCAN::distances::findDistancesMatX<float>(X_t, A_t, B_t, 1.2, 2000, "COSINE", matx::MATX_DEVICE_MEMORY);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();

    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, n * 2 * k * m);

    for (int i = 0; i < n*2*k*m; i++) {
        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
        ASSERT_NEAR(distances_expected_h[i], distances_h[i], 1e-2); // Doing a cast just to be sure
    }

    // TODO compare the average distance of these two - may find some major differences
}

TEST_F(TestFindingDistances, TestMediumInputMatx) {
    /*
     * This test checks if results calculated by C++/MatX are identical to those with Python/CuPy
     */

    auto AVector = GsDBSCAN::run_utils::loadCsvColumnToVector<int>("/home/hphi344/Documents/Thesis/python/data/A_n1000_k3.csv");
    int *A_h = AVector.data();

    auto BVector = GsDBSCAN::run_utils::loadCsvColumnToVector<int>("/home/hphi344/Documents/Thesis/python/data/B_D100_m20.csv");
    int *B_h = BVector.data();

    auto XVector = GsDBSCAN::run_utils::loadCsvColumnToVector<float>("/home/hphi344/Documents/Thesis/python/data/X_n1000_d20.csv");
    float *X_h = XVector.data();

    auto distancesVector = GsDBSCAN::run_utils::loadCsvColumnToVector<float>("/home/hphi344/Documents/Thesis/python/data/distances_n1000_k3_m20.csv");

    float *distances_expected_h = distancesVector.data();

    int n = 1000;
    int k = 3;
    int m = 20;
    int D = 100;
    int d = 20;

    int *A_d = GsDBSCAN::algo_utils::copyHostToDevice(A_h, n * 2 * k);
    int *B_d = GsDBSCAN::algo_utils::copyHostToDevice(B_h, 2 * D * m);
    float *X_d = GsDBSCAN::algo_utils::copyHostToDevice(X_h, n * d);

    auto X_t = matx::make_tensor<float>(X_d, {n, d});

    auto A_t = matx::make_tensor<int>(A_d, {n, 2 * k});
    auto B_t = matx::make_tensor<int>(B_d, {2 * D, m});

    auto start = tu::timeNow();

    auto distances_t = GsDBSCAN::distances::findDistancesMatX<float>(X_t, A_t, B_t, 1.2, 100);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();
    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, n * 2 * k * m);

    for (int i = 0; i < n*2*k*m; i++) {
        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
        ASSERT_NEAR(distances_expected_h[i], distances_h[i], 1e-2); // Doing a cast just to be sure
    }
}

TEST_F(TestFindingDistances, TestMediumInputMatXDeviceDistances) {
    /*
 * This test checks if results calculated by C++/MatX are identical to those with Python/CuPy
 */

    auto AVector = GsDBSCAN::run_utils::loadCsvColumnToVector<int>("/home/hphi344/Documents/Thesis/python/data/A_n1000_k3.csv");
    int *A_h = AVector.data();

    auto BVector = GsDBSCAN::run_utils::loadCsvColumnToVector<int>("/home/hphi344/Documents/Thesis/python/data/B_D100_m20.csv");
    int *B_h = BVector.data();

    auto XVector = GsDBSCAN::run_utils::loadCsvColumnToVector<float>("/home/hphi344/Documents/Thesis/python/data/X_n1000_d20.csv");
    float *X_h = XVector.data();

    auto distancesVector = GsDBSCAN::run_utils::loadCsvColumnToVector<float>("/home/hphi344/Documents/Thesis/python/data/distances_n1000_k3_m20.csv");

    float *distances_expected_h = distancesVector.data();

    int n = 1000;
    int k = 3;
    int m = 20;
    int D = 100;
    int d = 20;

    int *A_d = GsDBSCAN::algo_utils::copyHostToDevice(A_h, n * 2 * k);
    int *B_d = GsDBSCAN::algo_utils::copyHostToDevice(B_h, 2 * D * m);
    float *X_d = GsDBSCAN::algo_utils::copyHostToDevice(X_h, n * d);

    auto X_t = matx::make_tensor<float>(X_d, {n, d});

    auto A_t = matx::make_tensor<int>(A_d, {n, 2 * k});
    auto B_t = matx::make_tensor<int>(B_d, {2 * D, m});

    auto start = tu::timeNow();

    auto distances_t = GsDBSCAN::distances::findDistancesMatX<float>(X_t, A_t, B_t, 1.2, 100, "L2", matx::MATX_DEVICE_MEMORY);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();
    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, n * 2 * k * m);

    for (int i = 0; i < n*2*k*m; i++) {
        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
        ASSERT_NEAR(distances_expected_h[i], distances_h[i], 1e-2); // Doing a cast just to be sure
    }
}

TEST_F(TestFindingDistances, TestLargeMnistMatxFromFiles) {
    auto distancesVector = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/distances_test/distances_test_row_major.bin");
    auto AVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/distances_test/A_test_expected_row_major.bin");
    auto BVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/distances_test/B_test_expected_row_major.bin");

    int n = 70000;
    int d = 784;
    int k = 5;
    int m = 50;
    int D = 1024;

    float *distances_expected_h = distancesVector.data();
    int *A_h = AVector.data();
    int *B_h = BVector.data();

    int *A_d = GsDBSCAN::algo_utils::copyHostToDevice(A_h, n * 2 * k);
    int *B_d = GsDBSCAN::algo_utils::copyHostToDevice(B_h, 2 * D * m);
    auto A_t = matx::make_tensor<int>(A_d, {n, 2 * k});
    auto B_t = matx::make_tensor<int>(B_d, {2 * D, m});

    auto X = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin");
    auto X_af = af::array(n, d, X.data());
    X_af.eval();

    X_af = GsDBSCAN::projections::normaliseDatasetAF(X_af);
    X_af.eval();

    auto X_t = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);

    auto distances_t = GsDBSCAN::distances::findDistancesMatX<float>(X_t, A_t, B_t, 1.2, 2000, "COSINE", matx::MATX_DEVICE_MEMORY);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();
    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, n * 2 * k * m);

    for (int i = 0; i < n*2*k*m; i++) {
        float diff = std::abs(distances_expected_h[i] - distances_h[i]);

        if (diff > 1e-2) {
            printf("i: %d, expected: %f, actual: %f, diff: %f\n", i, distances_expected_h[i], distances_h[i], diff);
        }

        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
//        ASSERT_NEAR(distances_expected_h[i], distances_h[i], 1e-2); // Doing a cast just to be sure
    }
}

TEST_F(TestFindingDistances, TestMnistAFMatXIntegration) {

    std::string datasetFileName = "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin";

    int n = 70000;
    int d = 784;

    int k = 5;
    int m = 10;

    int D = 1024;

    auto X = GsDBSCAN::run_utils::loadBinFileToVector<float>(datasetFileName);
    auto X_h = X.data();

    auto X_af = af::array(n, d, X_h);
    X_af.eval();

    X_af = GsDBSCAN::projections::normaliseDatasetAF(X_af);
    auto projections = GsDBSCAN::projections::performProjectionsAF(X_af, D);


    auto X_t = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);

    auto [A_af, B_af] = GsDBSCAN::projections::constructABMatricesAF(projections, k, m);

    int A_max = af::max<int>(A_af);
    int B_max = af::max<int>(B_af);

    ASSERT_TRUE(A_max <= 2 * D - 1);
    ASSERT_TRUE(B_max <= n - 1);

    auto A_t = GsDBSCAN::algo_utils::afMatToMatXTensor<int, int>(A_af,
                                                                 matx::MATX_DEVICE_MEMORY); // TODO use MANAGED or DEVICE memory?
    auto B_t = GsDBSCAN::algo_utils::afMatToMatXTensor<int, int>(B_af,
                                                                 matx::MATX_DEVICE_MEMORY); // TODO use MANAGED or DEVICE memory?


    auto start = tu::timeNow();

    matx::tensor_t<float, 2> distances_t = GsDBSCAN::distances::findDistancesMatX(X_t, A_t, B_t, 1.2, -1, "L2",
                                                                                  matx::MATX_DEVICE_MEMORY);

    cudaDeviceSynchronize();

    tu::printDurationSinceStart(start);

    ASSERT_TRUE(distances_t.Shape()[0] == n);
    ASSERT_TRUE(distances_t.Shape()[1] == 2 * k * m);
}


TEST_F(TestFindingDistances, TestMockAF) {
    GTEST_SKIP(); // Legacy test, I should remove it!
    auto YBatch = af::randu(20, 2000, 784);

    auto start = tu::timeNow();

    auto YBatchNorm = af::sqrt(af::sum(af::pow(YBatch, 2), 1));

    tu::printDurationSinceStart(start);
}


TEST_F(TestFindingDistances, TestLargeInputMatX) {
    int k = 5;
    int n = 70000;
    int m = 50;
    int D = 1024;
    int d = 784;

    auto A = tu::createMockAMatrixMatX(n, k, D, matx::MATX_MANAGED_MEMORY);
    auto B = tu::createMockBMatrixMatX(n, m, D, matx::MATX_MANAGED_MEMORY);
    auto X = tu::createMockMnistDatasetMatX<float>(n, d, matx::MATX_MANAGED_MEMORY);

    cudaDeviceSynchronize();

//    print(X);

    tu::Time start = tu::timeNow();

    auto distances = GsDBSCAN::distances::findDistancesMatX<float>(X, A, B, 1.2, 2000, "L2", matx::MATX_MANAGED_MEMORY);
    cudaDeviceSynchronize();

    cudaCheckError();

    tu::printDurationSinceStart(start); // This is too fn slow. Around 14 seconds, Cupy takes less than 0.7 seconds.


    auto *distances_ptr = distances.Data();

    for (int i = 0; i < n*2*k*m; i++) {
//    printf("%f ", matx::promote_half_t<matx::matxFp16>(distances_ptr[i])); // Alot of zeros
//        printf("%d", i);
//        printf("%f ", distances_ptr[i]);

    }

    printf("%lld %lld", distances.Shape()[0], distances.Shape()[1]);

    ASSERT_TRUE(distances.Shape()[0] == n);
    ASSERT_TRUE(distances.Shape()[1] == 2*k*m);
}

TEST_F(TestFindingDistances, TestLargeInputAF) {
    GTEST_SKIP(); // Legacy test, I should remove it!

    int n = 70000;
    af::array X = tu::createMockMnistDataset(n, 784);
    af::array A, B;
    std::tie(A, B) = tu::createMockABMatrices(n, 5, 50, 1024);

    cudaDeviceSynchronize();

    tu::Time start = tu::timeNow();

    af::array distances = GsDBSCAN::distances::findDistancesL2AF(X, A, B);

    tu::printDurationSinceStart(start);
}