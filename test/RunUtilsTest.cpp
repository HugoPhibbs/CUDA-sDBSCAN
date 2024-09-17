//
// Created by hphi344 on 21/08/24.
//
#include "../include/pch.h"
#include "../include/gsDBSCAN/run_utils.h"
#include "../include/TestUtils.h"
#include "../include/gsDBSCAN/GsDBSCAN_Params.h"

#include <iostream>

namespace tu = testUtils;


class RunUtilsTest : public ::testing::Test {

};

class TestReadingImages : public RunUtilsTest {

};

//TEST_F(TestReadingImages, TestReadingImages) {
//    auto start = tu::timeNow();
//
//    auto vec = GsDBSCAN::run_utils::loadBinFileToVector<float>(
//            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin");
//
//    tu::printDurationSinceStart(start, "Reading MNIST images");
//
//    auto X_h = vec.data();
//    auto X_h_row_major = GsDBSCAN::algo_utils::colMajorToRowMajorMat(X_h, 70000, 784);
//
//    cv::Mat img(28, 28, CV_8UC1);
//}

class TestMainHelper : public RunUtilsTest {

};

TEST_F(TestMainHelper, TestNormally) {
    GsDBSCAN::GsDBSCAN_Params params = GsDBSCAN::GsDBSCAN_Params(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin",
            "",
            70000,
            784,
            1024,
            50,
            2,
            2000,
            0.11,
            "COSINE",
            true,
            true
        );

    params.useBatchClustering=false;
    params.verbose=true;

    std::cout << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    std::cout << "Number of clusters: " << numClusters << std::endl;

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