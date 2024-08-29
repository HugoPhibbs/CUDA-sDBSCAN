//
// Created by hphi344 on 21/08/24.
//

#include <gtest/gtest.h>
//#include <opencv4/opencv2/opencv.hpp>
#include "../include/gsDBSCAN/run_utils.h"
#include "../include/TestUtils.h"

namespace tu = testUtils;


class RunUtilsTest : public ::testing::Test {

};

class TestReadingImages: public RunUtilsTest {

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
    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin", 70000, 784, 1024, 100, 5, 50,
            0.11, 1.2, -1, "COSINE", 256);
}

TEST_F(TestMainHelper, TestExactDBSCAN) {
    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin", 70000, 784, 1024, 50, 1, 70000,
            0.11, 1.2, -1, "COSINE", 256);
}

TEST_F(TestMainHelper, TestWithCpuClustering) {
    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin", 70000, 784, 1024, 50, 5, 50,
            0.11, 1.2, -1, "COSINE", 256, true);
}

TEST_F(TestMainHelper, TestCpuClusteringk1m2000) {
    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin", 70000, 784, 1024, 50, 1, 2000,
            0.11, 1.2, -1, "COSINE", 256, true);

    std::cout<< "Num Clusters: "<< numClusters<<std::endl;
}

TEST_F(TestMainHelper, TestCpuClusteringk50m40) {
    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            "/home/hphi344/Documents/GS-DBSCAN-Analysis/data/mnist_images_col_major.bin", 70000, 784, 1024, 50, 50, 40,
            0.11, 1.2, -1, "COSINE", 256, true);

    std::cout<< "Num Clusters: "<< numClusters<<std::endl;
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