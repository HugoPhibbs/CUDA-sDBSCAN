//
// Created by hphi344 on 9/08/24.
//

#include <gtest/gtest.h>
#include "../../include/gsDBSCAN/GsDBSCAN.h"

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