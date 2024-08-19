//
// Created by hphi344 on 19/08/24.
//

#include <string>
#include <vector>

#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/gsDBSCAN/utils.h"

int main() {

    std::string XFilename = "/home/hphi344/Documents/Thesis/python/data/mnist_images_col_major.csv";

    std::vector<float> X_vec = GsDBSCAN::utils::loadCsvColumnToVector<float>(XFilename, 0);
    float *X_h = X_vec.data();

    int n = 70000;
    int d = 784;
    int k = 5;
    int m = 50;
    int D = 1024;

    float eps = 0.11;
    int minPts = 20;

    float *X_d = GsDBSCAN::utils::copyHostToDevice(X_h, n * d);

    auto [clusterLabels, typeLabels] = GsDBSCAN::performGsDbscan(
            X_d,
            n,
            d,
            D,
            minPts,
            k,
            m,
            eps
    );
}