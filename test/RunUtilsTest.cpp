//
// Created by hphi344 on 21/08/24.
//

#include <gtest/gtest.h>
#include "../include/gsDBSCAN/run_utils.h"
#include "../include/TestUtils.h"

namespace tu = testUtils;

class RunUtilsTest : public ::testing::Test {

};

class TestMainHelper : public RunUtilsTest {

};

TEST_F(TestMainHelper, TestNormally) {
    auto [clusterLabels, typeLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin", 70000, 784, 1024, 50, 5, 10,
            0.11, 1.2, -1, "L2", 256);
}

class TestReadMnist : public RunUtilsTest {

};

TEST_F(TestReadMnist, TestCSVNormally) {
    auto start = tu::timeNow();

    GsDBSCAN::run_utils::loadCsvColumnToVector<float>(
            "/home/hphi344/Documents/Thesis/python/data/mnist_images_col_major.csv", 0);

    tu::printDurationSinceStart(start, "Reading MNIST via csv");
}

TEST_F(TestReadMnist, TestBinNormally) {
    auto start = tu::timeNow();

    auto vec = GsDBSCAN::run_utils::loadBinFileToVector<float>(
            "/home/hphi344/Documents/Thesis/python/data/mnist_images_col_major.bin");

    tu::printDurationSinceStart(start, "Reading MNIST via binary");
}