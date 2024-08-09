//
// Created by hphi344 on 9/08/24.
//

#include <gtest/gtest.h>
#include <arrayfire.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"

namespace tu = testUtils;

class ClusteringTest : public ::testing::Test {

};

class TestConstructQueryVectorDegreeArray : public ClusteringTest {

};


TEST_F(TestConstructQueryVectorDegreeArray, TestSmallInput) {
    float distancesData[] = {
            0, 1, 2, 3,
            0, 2, 1, 0,
            1, 8, 9, 11,
            15, 2, 6, 7
    };

    af::array distances(4, 4, distancesData);

    float eps = 2.1;

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);

    float expectedData[] = {3, 4, 1, 1};

    af::array expected(1, 4, expectedData);

    ASSERT_TRUE(af::allTrue<bool>(expected == E));
}

TEST_F(TestConstructQueryVectorDegreeArray, TestMnist) {
    af::array distances = tu::createMockDistances();

    float eps = af::randu(1, 1).scalar<float>();

    tu::Time start = tu::timeNow();

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);

    E.eval();
    af::sync();

    tu::printDurationSinceStart(start);
}

class TestProcessQueryVectorDegreeArray : public ClusteringTest {

};

TEST_F(TestProcessQueryVectorDegreeArray, TestSmallInput) {

    float EData[] = {3, 4, 1, 1};
    af::array E(1, 4, EData);

    float VExpectedData[] = {0, 3, 7, 8};

    af::array VExpected(1, 4, VExpectedData);

    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    ASSERT_TRUE(af::allTrue<bool>(VExpected ==  V));
}

TEST_F(TestProcessQueryVectorDegreeArray, TestMnist) {
    af::array distances = tu::createMockDistances();

    float eps = af::randu(1, 1).scalar<float>();

    af::array E = GsDBSCAN::constructQueryVectorDegreeArray(distances, eps);

    tu::Time start = tu::timeNow();

    af::array V = GsDBSCAN::processQueryVectorDegreeArray(E);

    V.eval();
    af::sync();

    tu::printDurationSinceStart(start);
}