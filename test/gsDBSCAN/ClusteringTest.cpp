//
// Created by hphi344 on 9/08/24.
//

#include <gtest/gtest.h>
#include <arrayfire.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"
#include <cuda_runtime.h>

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

};

class TestSortingProjections : public ClusteringTest {

};

//TEST_F(TestSortingProjections, TestSortingProjectionsLargeInputMatX) {
//    // A perf test, doesn't unit test anything
//
//    int n = 20000;
//    int D = 1000;
//
//    auto mockProjections = matx::random<float>({n, D}, matx::UNIFORM) * 100;
//
//    mockProjections.run();
//
//    cudaDeviceSynchronize();
//
//    auto start = tu::timeNow();
//
//    matx::tensor_t<float, 2> projectionsSorted = matx::make_tensor<float>({n, D});
//
//    (projectionsSorted = matx::sort(mockProjections, matx::SORT_DIR_ASC)).run();
//
//    cudaDeviceSynchronize();
//
//    tu::printDurationSinceStart(start);
//
////    auto projectionsSortedT = matx::sort(matx::transpose(mockProjections), matx::SORT_DIR_ASC);
////
////    cudaDeviceSynchronize();
////
////    matx::print(projectionsSorted);
//
//
//    // TODO figure out why the sorting is so quick, frankly there is no way it is that quick
//};


class TestConstructQueryVectorDegreeArray : public ClusteringTest {

};

TEST_F(TestConstructQueryVectorDegreeArray, TestSmallInputMatX) {
    float distancesData[] = {
            0, 1, 2, 3,
            0, 2, 1, 0,
            1, 8, 9, 11,
            15, 2, 6, 7
    };

    auto distancesData_d = GsDBSCAN::utils::copyHostToDevice(distancesData, 16, true);

    matx::tensor_t<float, 2> distances_t = matx::make_tensor<float>(distancesData_d, {4, 4}, matx::MATX_MANAGED_MEMORY);

    auto degArray_t = GsDBSCAN::clustering::constructQueryVectorDegreeArrayMatx<float>(distances_t, 2.1, matx::MATX_MANAGED_MEMORY); // TODO fix me!

    auto degArray_d = degArray_t.Data();

    int expectedData[] = {3, 4, 1, 1};

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(degArray_d[i], expectedData[i]);
    }
}


TEST_F(TestConstructQueryVectorDegreeArray, TestSmallInputAF) {
    float distancesData[] = {
            0, 1, 2, 3,
            0, 2, 1, 0,
            1, 8, 9, 11,
            15, 2, 6, 7
    };

    af::array distances(4, 4, distancesData);

    float eps = 2.1;

    af::array E = GsDBSCAN::clustering::constructQueryVectorDegreeArray(distances, eps);

    float expectedData[] = {3, 4, 1, 1};

    af::array expected(1, 4, expectedData);

    ASSERT_TRUE(af::allTrue<bool>(expected == E));
}

TEST_F(TestConstructQueryVectorDegreeArray, TestMnist) {
    af::array distances = tu::createMockDistances();

    float eps = af::randu(1, 1).scalar<float>();

    tu::Time start = tu::timeNow();

    af::array E = GsDBSCAN::clustering::constructQueryVectorDegreeArray(distances, eps);

    E.eval();
    af::sync();

    tu::printDurationSinceStart(start);
}

class TestProcessQueryVectorDegreeArray : public ClusteringTest {

};

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInputMatX) {
    int EData[] = {3, 4, 1, 1};

    auto EData_d = GsDBSCAN::utils::copyHostToDevice(EData, 4, true);

    auto V = GsDBSCAN::clustering::processQueryVectorDegreeArrayThrust(EData_d, 4);

    cudaDeviceSynchronize();

//    auto VData = V.Data(); // TODO why can't i get a pointer to the data?
}

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInputThrust) {

    int degArray[] = {3, 4, 1, 1};

    int *degArray_d = GsDBSCAN::utils::copyHostToDevice(degArray, 4, true);

    int *startIdxArray_d = GsDBSCAN::clustering::processQueryVectorDegreeArrayThrust(degArray_d, 4);

    int *startIdxArray_h = GsDBSCAN::utils::copyDeviceToHost(startIdxArray_d, 4);

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

    auto distancesData_d = GsDBSCAN::utils::copyHostToDevice(distancesData, 16, false);

    matx::tensor_t<float, 2> distances_t = matx::make_tensor<float>(distancesData_d, {4, 4}, matx::MATX_DEVICE_MEMORY);

    auto degArray_t = GsDBSCAN::clustering::constructQueryVectorDegreeArrayMatx<float>(distances_t, 2.1, matx::MATX_DEVICE_MEMORY);

    auto degArray_d = degArray_t.Data();

    int *startIdxArray_d = GsDBSCAN::clustering::processQueryVectorDegreeArrayThrust(degArray_d, 4);

    int *startIdxArray_h = GsDBSCAN::utils::copyDeviceToHost(startIdxArray_d, 4);

    int expectedData[] = {0, 3, 7, 8};

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(expectedData[i], startIdxArray_h[i]);
    }
}

TEST_F(TestProcessQueryVectorDegreeArray, TestMnist) {
    af::array distances = tu::createMockDistances();

    float eps = af::randu(1, 1).scalar<float>();

    af::array E = GsDBSCAN::clustering::constructQueryVectorDegreeArray(distances, eps);

    tu::Time start = tu::timeNow();

    af::array V = GsDBSCAN::clustering::processQueryVectorDegreeArray(E);

    V.eval();
    af::sync();

    tu::printDurationSinceStart(start);
}

class TestFormingClusters : public ClusteringTest {

};

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

    int *adjacencyList_d = GsDBSCAN::utils::copyHostToDevice(adjacencyList_h, 16, true);
    int *degArray_d = GsDBSCAN::utils::copyHostToDevice(degArray_h, n, true);
    int *startIdxArray_d = GsDBSCAN::utils::copyHostToDevice(startIdxArray_h, n, true);

    auto start = tu::timeNow();

    auto [clusterLabels_h, typeLabels_h] = GsDBSCAN::clustering::formClusters(adjacencyList_d, degArray_d, startIdxArray_d, n, minPts);

    tu::printDurationSinceStart(start);

    int clusterLabelsExpected_h[12] = {0, 0, 0, 0, -1, 1, 1, 1, -1, 1, -1, -1};
    int typeLabelsExpected_h[12] = {0, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, -1};

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(clusterLabelsExpected_h[i], clusterLabels_h[i]);
        ASSERT_EQ(typeLabelsExpected_h[i], typeLabels_h[i]);
    }

}

TEST_F(TestFormingClusters, TestSmallInputB) {
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

    int *adjacencyList_d = GsDBSCAN::utils::copyHostToDevice(adjacencyList_h, 16, true);
    int *degArray_d = GsDBSCAN::utils::copyHostToDevice(degArray_h, n, true);
    int *startIdxArray_d = GsDBSCAN::utils::copyHostToDevice(startIdxArray_h, n, true);

    int clusterLabelsExpected_h[12] = {0, 0, 0, 0, -1, 1, 1, 1, -1, 1, -1, -1};
    int typeLabelsExpected_h[12] = {0, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, -1};

    auto [clusterLabels_h_small_block, typeLabels_h_small_block] = GsDBSCAN::clustering::formClusters(adjacencyList_d, degArray_d, startIdxArray_d, n, minPts, 3);

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(clusterLabelsExpected_h[i], clusterLabels_h_small_block[i]);
        ASSERT_EQ(typeLabelsExpected_h[i], typeLabels_h_small_block[i]);
    }
}