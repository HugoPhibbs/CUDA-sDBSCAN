//
// Created by hphi344 on 21/08/24.
//
#include "../include/pch.h"
#include "../include/gsDBSCAN/run_utils.h"
#include "../include/TestUtils.h"
#include "../include/gsDBSCAN/GsDBSCAN_Params.h"
#include <gtest/gtest.h>

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
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major.bin",
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

    params.useBatchClustering = false;
    params.verbose = true;

    std::cout << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    std::cout << "Number of clusters: " << numClusters << std::endl;

}

TEST_F(TestMainHelper, TestNormallyF16) {
    GsDBSCAN::GsDBSCAN_Params params = GsDBSCAN::GsDBSCAN_Params(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist/mnist_images_row_major_f16.bin",
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

    params.useBatchClustering = false;
    params.verbose = true;
    params.datasetDType = "f16";

    std::cout << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    std::cout << "Number of clusters: " << numClusters << std::endl;

}

TEST_F(TestMainHelper, TestMnist8m1MF16) {
    GsDBSCAN::GsDBSCAN_Params params = GsDBSCAN::GsDBSCAN_Params(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/data/mnist8m_f16_sample_1000000.bin",
            "",
            1000000,
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

    params.useBatchClustering = true;
    params.verbose = true;
    params.datasetDType = "f16";

    std::cout << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    std::cout << "Number of clusters: " << numClusters << std::endl;

    ASSERT_TRUE(numClusters > 3);
}

TEST_F(TestMainHelper, TestPortoF32) {
    GsDBSCAN::GsDBSCAN_Params params = GsDBSCAN::GsDBSCAN_Params(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/spatial_datasets/porto_s8_f32.bin",
            "",
            10427415,
            2,
            1024,
            8,
            6,
            160,
            0.008,
            "L2",
            true,
            false
    );

    params.useBatchNorm = true;
    params.useBatchClustering = true;
    params.verbose = true;
    params.datasetDType = "f32";
    params.ABatchSize = 100000;
    params.BBatchSize = 32;
    params.miniBatchSize = 100000;

    std::cout << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    std::cout << "Number of clusters: " << numClusters << std::endl;
}

TEST_F(TestMainHelper, TestMnist8M3M) {
    GsDBSCAN::GsDBSCAN_Params params = GsDBSCAN::GsDBSCAN_Params(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist8m/samples/f16/mnist8m_sample_n3000000_f16.bin",
            "",
            3000000,
            784,
            1024,
            50,
            10,
            50,
            5.3,
            "L2",
            true,
            false
    );

    params.useBatchClustering = true;
    params.verbose = true;
    params.datasetDType = "f16";
    params.ABatchSize = 10000;
    params.BBatchSize = 28;
    params.miniBatchSize = 10000;

    std::cout << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    std::cout << "Number of clusters: " << numClusters << std::endl;
}


class TestReadMnist : public RunUtilsTest {

};


TEST_F(TestReadMnist, TestBinNormally) {
    auto start = tu::timeNow();

    auto vec = GsDBSCAN::run_utils::loadBinFileToVector<float>(
            "/home/hphi344/Documents/Thesis/python/data/mnist/mnist_images_col_major.bin");

    tu::printDurationSinceStart(start, "Reading MNIST via binary");
}

