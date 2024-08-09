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
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

#include "../../include/rapidcsv.h"
#include "../../include/gsDBSCAN/GsDBSCAN.h"
#include "../../include/TestUtils.h"

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}