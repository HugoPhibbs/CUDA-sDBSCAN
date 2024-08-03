//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include "Header.h"
#include "Utilities.h"
#include <arrayfire.h>
#include <cuda_runtime.h>
#include <af/cuda.h>
#include <arrayfire.h>
#include <matx.h>

class GsDBSCAN {
private :
    af::array X;
    int minPts;
    int k;
    int m;
    float eps;
    bool skip_pre_checks;
    int n;
    int d;
    int D;
    float sigma;
    int seed;
    string distanceMetric;
    bool clusterNoise;
    int fhtDim;
    float batchAlpha;
    int nRotate;

    boost::dynamic_bitset<> bitHD3;

public:
    // Constructor if needed
    GsDBSCAN(const af::array &X, int D, int minPts, int k, int m, float eps, bool skip_pre_checks, float sigma, int seed, string distanceMetric, float batchAlpha, int fhtDim, int nRotate, bool clusterNoise);

    // Methods corresponding to the functions
    void performGsDbscan();

    void static preChecks(af::array &X, int D, int minPts, int k, int m, float eps);

    std::tuple<af::array, af::array> static preProcessing(af::array &X, int D, int k, int m);

    af::array static randomProjections(af::array &X, boost::dynamic_bitset<> bitHD3, int D, int k, int m, string distanceMetric, float sigma, int seed, int fhtDim, int nRotate);

    std::tuple<af::array, af::array> static constructABMatrices(const af::array& projections, int k, int m);

    af::array static findDistances(af::array &X, af::array &A, af::array &B, float alpha = 1.2);

    matx::tensor_t<matx::matxFp16, 2> static findDistancesMatX(matx::tensor_t<matx::matxFp16, 2> &X_t, matx::tensor_t<int, 2> &A_t, matx::tensor_t<int, 2> &B_t, float alpha = 1.2, int batchSize=-1);

    int static findDistanceBatchSize(float alpha, int n, int d, int k, int m);

    static af::array assembleAdjacencyList(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B, float eps, int blockSize=1024);

    af::array static constructQueryVectorDegreeArray(af::array &distances, float eps);

    af::array static processQueryVectorDegreeArray(af::array &E);

    tuple<vector<int>, int> static formClusters(af::array &adjacencyList, af::array &V, af::array &E, int n, int minPts, bool clusterNoise);

    cudaStream_t static getAfCudaStream();

    // Destructor if needed
    ~GsDBSCAN() = default;
};

#endif //DBSCANCEOS_GSDBSCAN_H
