//
// Created by hphi344 on 21/08/24.
//

#include <gtest/gtest.h>
#include <../../include/gsDBSCAN/run_utils.h>

class gs_mainTest: public ::testing::Test  {

};

class TestMainHelper : public gs_mainTest {

};

TEST_F(TestMainHelper, TestNormally) {
    auto [clusterLabels, typeLabels, times] = GsDBSCAN::run_utils::main_helper("/home/hphi344/Documents/Thesis/python/data/mnist_images_col_major.bin", 70000, 784, 1024, 50, 5, 10, 0.11, 1.2, "COSINE", 256);
}
