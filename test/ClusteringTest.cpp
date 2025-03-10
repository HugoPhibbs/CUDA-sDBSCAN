//
// Created by hphi344 on 9/08/24.
//

#include "../include/pch.h"
#include <gtest/gtest.h>
#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/TestUtils.h"
#include "../include/gsDBSCAN/algo_utils.h"
#include "../include/gsDBSCAN/run_utils.h"
#include <cmath>

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


class ClusteringTest : public ::testing::Test {
protected:

    template <typename T>
    void assertArrayEqual(T* array1, T* array2, size_t size, T eps=1e-6) {
        for (size_t i = 0; i < size; ++i) {
            ASSERT_NEAR(array1[i], array2[i], eps);
        }
    }
};

class TestSortingProjections : public ClusteringTest {

};

class TestConstructQueryVectorDegreeArray : public ClusteringTest {

};

TEST_F(TestConstructQueryVectorDegreeArray, TestSmallInputMatX) {
    float distancesData[] = {
            0, 1, 2, 3, 5,
            0, 2, 1, 0, 2,
            1, 8, 9, 11, 0,
            15, 2, 6, 7, 1
    };

    auto distancesData_d = GsDBSCAN::algo_utils::copyHostToDevice(distancesData, 20, true);

    matx::tensor_t<float, 2> distances_t = matx::make_tensor<float>(distancesData_d, {4, 5}, matx::MATX_MANAGED_MEMORY);

    auto degArray_d = GsDBSCAN::clustering::constructQueryVectorDegreeArrayMatx<float>(distances_t, 2.1, "L2", matx::MATX_MANAGED_MEMORY);

    int expectedData[] = {3, 5, 2, 2};

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(degArray_d[i], expectedData[i]);
    }
}

TEST_F(TestConstructQueryVectorDegreeArray, TestMnistAgainstPython) {
    float eps = 1-0.11;

    int n = 70000;
    int numCandidates = 2*5*50; // 2 * k * m

    auto distancesData = GsDBSCAN::run_utils::loadBinFileToVector<float>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/distances_cosine_row_major.bin");

    auto distancesData_d = GsDBSCAN::algo_utils::copyHostToDevice(distancesData.data(), distancesData.size(), true);

    matx::tensor_t<float, 2> distances_t = matx::make_tensor<float>(distancesData_d, {n, numCandidates}, matx::MATX_MANAGED_MEMORY);

    auto degArray_d = GsDBSCAN::clustering::constructQueryVectorDegreeArrayMatx<float>(distances_t, eps, "COSINE", matx::MATX_MANAGED_MEMORY);

    auto degArray_h = GsDBSCAN::algo_utils::copyDeviceToHost(degArray_d, n);

    auto degArrayExpected = GsDBSCAN::run_utils::loadBinFileToVector<int>("/home/hphi344/Documents/GS-DBSCAN-Analysis/data/complete_test/deg_array_cosine.bin");

    int numMismatch = 0;

    for (int i = 0; i < n; i++) {
        if (degArray_h[i] != degArrayExpected[i]) {
            numMismatch ++;
        }
    }

    std::cout<< "Num mismatch: " <<numMismatch<<std::endl;

    assertArrayEqual(degArrayExpected.data(), degArray_h, n);
}

class TestProcessQueryVectorDegreeArray : public ClusteringTest {

};

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInputThrust) {

    int degArray[] = {3, 4, 1, 1};

    int *degArray_d = GsDBSCAN::algo_utils::copyHostToDevice(degArray, 4, true);

    int *startIdxArray_d = GsDBSCAN::clustering::constructStartIdxArray(degArray_d, 4);

    int *startIdxArray_h = GsDBSCAN::algo_utils::copyDeviceToHost(startIdxArray_d, 4);

    int expectedData[] = {0, 3, 7, 8};

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(expectedData[i], startIdxArray_h[i]);
    }
}

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInputIntegrationThrust) {
    float distancesData[] = {
            0, 1, 2, 3,
            0, 2, 1, 0,
            1, 8, 9, 11,
            15, 2, 6, 7
    };

    auto distancesData_d = GsDBSCAN::algo_utils::copyHostToDevice(distancesData, 16, false);

    matx::tensor_t<float, 2> distances_t = matx::make_tensor<float>(distancesData_d, {4, 4}, matx::MATX_DEVICE_MEMORY);

    auto degArray_d = GsDBSCAN::clustering::constructQueryVectorDegreeArrayMatx<float>(distances_t, 2.1, "L2", matx::MATX_DEVICE_MEMORY);

    int *startIdxArray_d = GsDBSCAN::clustering::constructStartIdxArray(degArray_d, 4);

    int *startIdxArray_h = GsDBSCAN::algo_utils::copyDeviceToHost(startIdxArray_d, 4);

    int expectedData[] = {0, 3, 7, 8};

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(expectedData[i], startIdxArray_h[i]);
    }
}

class TestCreatingAdjacencyList : public ClusteringTest {

};

TEST_F(TestCreatingAdjacencyList, TestSmallInput) {
    int degArray_h[12] = {1, 3, 1, 1, 0, 3, 2, 2, 0, 3, 1, 1};
    int startIdxArray_h[12] = {0, 1, 4, 5, 6, 6, 9, 11, 13, 13, 16, 17};

    // Assume eps = 3
    // Just take 4 candidate vectors, this will do for the simple case

    float *distances = {

    };

    // TODO
}

TEST_F(TestCreatingAdjacencyList, TestSmallInputCase2) {
    // This example was taken from my python implementation

    int A[10] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };

    auto A_d = GsDBSCAN::algo_utils::copyHostToDevice(A, 10, false);

    // Flattened integer array for B
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

    auto B_d = GsDBSCAN::algo_utils::copyHostToDevice(B, 30, false);


    float distancesSquared[] = {
            11.0f,  5.0f, 14.0f, 11.0f,  0.0f,  5.0f,
            9.0f,  0.0f, 11.0f,  0.0f, 14.0f, 11.0f,
            5.0f,  0.0f,  5.0f,  5.0f,  8.0f, 14.0f,
            9.0f,  5.0f,  0.0f,  0.0f,  9.0f,  5.0f,
            9.0f,  6.0f,  5.0f,  5.0f,  0.0f,  6.0f
    };

    // Create a new array for the square rooted distances
    float distances[30];

    // Loop through the distances array and take the square root of each element
    for (size_t i = 0; i < 30; ++i) {
        distances[i] = std::sqrt(distancesSquared[i]);
    }

    auto distances_d = GsDBSCAN::algo_utils::copyHostToDevice(distances, 30, false);

    int degArray[5] = {3, 2, 4, 4, 3};
    int startIdxArray[5] = {0, 3, 5, 9, 13};

    int *degArray_d = GsDBSCAN::algo_utils::copyHostToDevice(degArray, 5, false);
    int *startIdxArray_d = GsDBSCAN::algo_utils::copyHostToDevice(startIdxArray, 5, false);

    int n = 5;
    int k = 1;
    int m = 3;
    float eps = std::sqrt(5.1);

    auto [adjacencyList_d, adjacencyList_size] = GsDBSCAN::clustering::constructAdjacencyList(distances_d, degArray_d, startIdxArray_d, A_d, B_d, n, k, m, eps, 128, "L2");

    auto adjacencyList_h = GsDBSCAN::algo_utils::copyDeviceToHost(adjacencyList_d, adjacencyList_size);

    int adjacencyListExpected_h[18] {
            2, 0, 2,
            1, 1,
            0, 2, 3, 0,
            2, 3, 3, 2,
            0, 0, 4
    };

    for (int i = 0; i < adjacencyList_size; i++) {
        ASSERT_EQ(adjacencyListExpected_h[i], adjacencyList_h[i]);
    }
}

TEST_F(TestCreatingAdjacencyList, TestSmallInputCase2WithBatching) {
    int A[10] = {
            0, 3,
            2, 5,
            4, 1,
            0, 7,
            2, 1
    };

    auto A_d = GsDBSCAN::algo_utils::copyHostToDevice(A, 10, false);

    // Flattened integer array for B
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

    auto B_d = GsDBSCAN::algo_utils::copyHostToDevice(B, 30, false);


    float distancesSquared[] = {
            11.0f,  5.0f, 14.0f, 11.0f,  0.0f,  5.0f,
            9.0f,  0.0f, 11.0f,  0.0f, 14.0f, 11.0f,
            5.0f,  0.0f,  5.0f,  5.0f,  8.0f, 14.0f,
            9.0f,  5.0f,  0.0f,  0.0f,  9.0f,  5.0f,
            9.0f,  6.0f,  5.0f,  5.0f,  0.0f,  6.0f
    };

    // Create a new array for the square rooted distances
    float distances[30];

    // Loop through the distances array and take the square root of each element
    for (size_t i = 0; i < 30; ++i) {
        distances[i] = std::sqrt(distancesSquared[i]);
    }

    auto distances_d = GsDBSCAN::algo_utils::copyHostToDevice(distances, 30, false);

    int degArray[5] = {3, 2, 4, 4, 3};
    int startIdxArray[5] = {0, 3, 5, 9, 13};

    int *degArray_d = GsDBSCAN::algo_utils::copyHostToDevice(degArray, 5, false);
    int *startIdxArray_d = GsDBSCAN::algo_utils::copyHostToDevice(startIdxArray, 5, false);

    int n = 5;
    int k = 1;
    int m = 3;
    float eps = std::sqrt(5.1);

    auto [adjacencyList_d, adjacencyList_size] = GsDBSCAN::clustering::constructAdjacencyList(distances_d, degArray_d, startIdxArray_d, A_d, B_d, n, k, m, eps, 128, "L2");

    auto adjacencyList_h = GsDBSCAN::algo_utils::copyDeviceToHost(adjacencyList_d, adjacencyList_size);

    int adjacencyListExpected_h[18] {
            2, 0, 2,
            1, 1,
            0, 2, 3, 0,
            2, 3, 3, 2,
            0, 0, 4
    };

    for (int i = 0; i < adjacencyList_size; i++) {
        ASSERT_EQ(adjacencyListExpected_h[i], adjacencyList_h[i]);
    }
}

class TestFormingClusters : public ClusteringTest {

};

TEST_F(TestFormingClusters, TestSmallInputIntegration) {

}

TEST_F(TestFormingClusters, TestSmallInput) {
    // A simple case I came up with from a sketch

    int n = 12;
    int minPts = 3;

    int adjacencyList_h[18] = {
            1,
            0, 2, 3,
            1,
            1,
            9, 6, 7,
            5, 9,
            9, 5,
            5, 7, 6,
            11,
            10
    };

    int degArray_h[12] = {1, 3, 1, 1, 0, 3, 2, 2, 0, 3, 1, 1};
    int startIdxArray_h[12] = {0, 1, 4, 5, 6, 6, 9, 11, 13, 13, 16, 17};

    int *adjacencyList_d = GsDBSCAN::algo_utils::copyHostToDevice(adjacencyList_h, 18, true);
    int *degArray_d = GsDBSCAN::algo_utils::copyHostToDevice(degArray_h, n, true);
    int *startIdxArray_d = GsDBSCAN::algo_utils::copyHostToDevice(startIdxArray_h, n, true);

    auto start = tu::timeNow();

    auto [clusterLabels_h, typeLabels_h, numClusters] = GsDBSCAN::clustering::formClusters(adjacencyList_d, degArray_d, startIdxArray_d, n, minPts, 256);

    tu::printDurationSinceStart(start);

    int clusterLabelsExpected_h[12] = {0, 0, 0, 0, -1, 1, 1, 1, -1, 1, -1, -1};
    int typeLabelsExpected_h[12] = {0, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, -1};

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(clusterLabelsExpected_h[i], clusterLabels_h[i]);
        ASSERT_EQ(typeLabelsExpected_h[i], typeLabels_h[i]);
    }

    ASSERT_EQ(numClusters, 2);
}

TEST_F(TestFormingClusters, TestSmallInputWithBatches) {
    // A simple case I came up with from a sketch

    int n = 12;
    int minPts = 3;

    int adjacencyList_h[18] = {
            1,
            0, 2, 3,
            1,
            1,
            9, 6, 7,
            5, 9,
            9, 5,
            5, 7, 6,
            11,
            10
    };

    int degArray_h[12] = {1, 3, 1, 1, 0, 3, 2, 2, 0, 3, 1, 1};
    int startIdxArray_h[12] = {0, 1, 4, 5, 6, 6, 9, 11, 13, 13, 16, 17};

    int *adjacencyList_d = GsDBSCAN::algo_utils::copyHostToDevice(adjacencyList_h, 18, true);
    int *degArray_d = GsDBSCAN::algo_utils::copyHostToDevice(degArray_h, n, true);
    int *startIdxArray_d = GsDBSCAN::algo_utils::copyHostToDevice(startIdxArray_h, n, true);

    int clusterLabelsExpected_h[12] = {0, 0, 0, 0, -1, 1, 1, 1, -1, 1, -1, -1};
    int typeLabelsExpected_h[12] = {0, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, -1};

    auto [clusterLabels_h_small_block, typeLabels_h_small_block, numClusters] = GsDBSCAN::clustering::formClusters(adjacencyList_d, degArray_d, startIdxArray_d, n, minPts, 3);

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(clusterLabelsExpected_h[i], clusterLabels_h_small_block[i]);
        ASSERT_EQ(typeLabelsExpected_h[i], typeLabels_h_small_block[i]);
    }

    ASSERT_EQ(numClusters, 2);
}

//TEST_F(TestFormingClusters, TestSmallInputCpu) {
//    int n = 12;
//    int minPts = 3;
//
//    int adjacencyList_h[18] = {
//            1,
//            0, 2, 3,
//            1,
//            1,
//            9, 6, 7,
//            5, 9,
//            9, 5,
//            5, 7, 6,
//            11,
//            10
//    };
//
//    int degArray_h[12] = {1, 3, 1, 1, 0, 3, 2, 2, 0, 3, 1, 1};
//    int startIdxArray_h[12] = {0, 1, 4, 5, 6, 6, 9, 11, 13, 13, 16, 17};
//
//    int *adjacencyList_d = GsDBSCAN::algo_utils::copyHostToDevice(adjacencyList_h, 18, true);
//    int *degArray_d = GsDBSCAN::algo_utils::copyHostToDevice(degArray_h, n, true);
//    int *startIdxArray_d = GsDBSCAN::algo_utils::copyHostToDevice(startIdxArray_h, n, true);
//
//    int clusterLabelsExpected_h[12] = {0, 0, 0, 0, -1, 1, 1, 1, -1, 1, -1, -1};
//
//    auto [neighbourhoodMatrix, corePoints] = GsDBSCAN::clustering::processAdjacencyListCpu(adjacencyList_d, degArray_d, startIdxArray_d, n, 18, minPts);
//
//    auto [clusterLabels_h, numClusters] = GsDBSCAN::clustering::formClustersCPU(neighbourhoodMatrix,
//                                                                                corePoints, n);
//
//    for (int i = 0; i < n; i++) {
//        ASSERT_EQ(clusterLabelsExpected_h[i], clusterLabels_h[i]);
//    }
//
//    ASSERT_EQ(numClusters, 2);
//}