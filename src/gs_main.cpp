//
// Created by hphi344 on 19/08/24.
//

#include <string>
#include <vector>

#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "cuda_runtime.h"
#include "../include/gsDBSCAN/utils.h"
#include <arrayfire.h>

int main() {

    std::string XFilename = "/home/hphi344/Documents/Thesis/python/data/mnist_images_col_major.csv";
//
//    std::vector<float> X_vec = GsDBSCAN::utils::loadCsvColumnToVector<float>(XFilename, 0);
//    float *X_h = X_vec.data();

    int n = 1000;
    int d = 200;
    int k = 5;
    int m = 20;
    int D = 200;

    auto X_af = af::randu(1000, 200, f32);

    X_af.eval();

    float eps = 0.11;
    int minPts = 20;

//    float *X_d = GsDBSCAN::utils::copyHostToDevice(X_h, n * d);

    auto [clusterLabels, typeLabels] = GsDBSCAN::performGsDbscan(
            X_af.device<float>(),
            n,
            d,
            D,
            minPts,
            k,
            m,
            eps
    );

//    cudaFree(X_d);


    X_af.unlock();

    return 0;
}