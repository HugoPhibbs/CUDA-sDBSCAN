//
// Created by hphi344 on 10/05/24.
//
#include <gtest/gtest.h>
#include <matx.h>
#include <chrono>
#include <arrayfire.h>
#include <af/cuda.h>
#include <vector>
#include <string>
#include <cmath>

#include "../include/lib_include/rapidcsv.h"
#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/TestUtils.h"

namespace tu = testUtils;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}