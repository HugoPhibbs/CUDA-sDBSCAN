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

class TestSortingProjections : public ClusteringTest{

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

    auto distancesData_d = GsDBSCAN::utils::hostToManagedArray(distancesData, 16);

    matx::tensor_t<float, 2> distances_t = matx::make_tensor<float>(distancesData_d, {4, 4});

    int* E = GsDBSCAN::clustering::constructQueryVectorDegreeArrayMatx<float>(distances_t, 2.1); // TODO fix me!

    cudaDeviceSynchronize();

    // As as side reference,

    // TODO fix why all this nonsense doesn't work
//
//    matx::tensor_t<float, 2> E_adj = matx::reshape(E, {1, 4});
//
//    float * EData = E_adj.Data();
//
//    float expectedData[] = {3, 4, 1, 1};
//
//    for (int i = 0; i < 4; i++) {
//        ASSERT_EQ(EData[i], expectedData[i]);
//    }
}


TEST_F(TestConstructQueryVectorDegreeArray, TestSmallInput) {
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

    auto EData_d = GsDBSCAN::utils::hostToManagedArray(EData, 4);

    auto V = GsDBSCAN::clustering::processQueryVectorDegreeArrayThrust(EData_d, 4);

    cudaDeviceSynchronize();

//    auto VData = V.Data(); // TODO why can't i get a pointer to the data?
}

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInput) {

    float EData[] = {3, 4, 1, 1};
    af::array E(1, 4, EData);

    float VExpectedData[] = {0, 3, 7, 8};

    af::array VExpected(1, 4, VExpectedData);

    af::array V = GsDBSCAN::clustering::processQueryVectorDegreeArray(E);

    ASSERT_TRUE(af::allTrue<bool>(VExpected ==  V));
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