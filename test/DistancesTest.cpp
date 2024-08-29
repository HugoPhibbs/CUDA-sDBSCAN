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

    std::cout<<distances<<std::endl;

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

TEST_F(TestFindingDistances, TestLargeMnistMatxFromFilesCosine) {
    auto distancesVector = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/distances_cosine_row_major.bin");
    auto AVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/A_row_major.bin");
    auto BVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/B_row_major.bin");

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

    auto X = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/mnist_images_col_major.bin");
    auto X_af = af::array(n, d, X.data());
    X_af.eval();

    X_af = GsDBSCAN::projections::normaliseDatasetAF(X_af);
    X_af.eval();

    auto X_t = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);

    auto distances_t = GsDBSCAN::distances::findDistancesMatX<float>(X_t, A_t, B_t, 1.2, 250, "COSINE", matx::MATX_DEVICE_MEMORY);

    cudaDeviceSynchronize();

    auto distances_d = distances_t.Data();
    auto distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, n * 2 * k * m);

    int numDifferent = 0;

    for (int i = 0; i < n*2*k*m; i++) {
        float diff = std::abs(distances_expected_h[i] - distances_h[i]);

        if (diff > 1e-2) {
            numDifferent++;
        }

        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
//        ASSERT_NEAR(distances_expected_h[i], distances_h[i], 1e-2); // Doing a cast just to be sure
    }

    std::cout << "Num Different: " << numDifferent << std::endl;

    print(matx::slice(distances_t, {0, 0}, {10, 10}));
    auto average_h = std::accumulate(distances_h, distances_h + n*2*k*m, 0.0) / (n*2*k*m);
    auto average_expected_h = std::accumulate(distances_expected_h, distances_expected_h + n*2*k*m, 0.0) / (n*2*k*m);

    std::cout << "Average: actual:" << average_h << " expected:" << average_expected_h << std::endl;

    assertArrayEqual(distances_expected_h, distances_h, n*2*k*m, (float) 1e-2);
}

TEST_F(TestFindingDistances, TestLargeMnistMatxFromFilesL2) {
    auto distancesVector = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/distances_l2_row_major.bin");
    auto AVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/A_l2_row_major.bin");
    auto BVector = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/B_l2_row_major.bin");

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

    auto X = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/mnist_images_col_major.bin");
    auto X_af = af::array(n, d, X.data());
    X_af.eval();

    X_af = GsDBSCAN::projections::normaliseDatasetAF(X_af);
    X_af.eval();

    auto X_t = GsDBSCAN::algo_utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);

    auto distances_t = GsDBSCAN::distances::findDistancesMatX<float>(X_t, A_t, B_t, 1.2, 2000, "L2", matx::MATX_DEVICE_MEMORY);

    cudaDeviceSynchronize();

    auto *distances_d = distances_t.Data();
    auto *distances_h = GsDBSCAN::algo_utils::copyDeviceToHost(distances_d, n * 2 * k * m);

    int numDifferent = 0;

    for (int i = 0; i < n*2*k*m; i++) {
        float diff = std::abs(distances_expected_h[i] - distances_h[i]);

        if (diff > 1e-2) {
            numDifferent++;
        }

        // Python and CPP produce *slightly* different results. Hence, why I use a 1e-2 tolerance
//        ASSERT_NEAR(distances_expected_h[i], distances_h[i], 1e-2); // Doing a cast just to be sure
    }

    std::cout << "Num Different: " << numDifferent << std::endl;

    auto average_h = std::accumulate(distances_h, distances_h + n*2*k*m, 0.0) / (n*2*k*m); // TODO is this an appropriate number?
    auto average_expected_h = std::accumulate(distances_expected_h, distances_expected_h + n*2*k*m, 0.0) / (n*2*k*m);

    std::cout << "Average: actual:" << average_h << " expected:" << average_expected_h << std::endl;

    assertArrayEqual(distances_expected_h, distances_h, n*2*k*m, (float) 1e-2);
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