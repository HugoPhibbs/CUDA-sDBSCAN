//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include "Header.h"
#include "Utilities.h"
#include <Eigen/Dense>
#include <arrayfire.h>

class GsDBSCAN {
private :
    af::array X;
    af::array D;
    int minPts;
    int k;
    int m;
    float eps;
    bool skip_pre_checks;
    int n;
    int d;

public:
    // Constructor if needed
    GsDBSCAN(const af::array &X, const af::array &D, int minPts, int k, int m, float eps, bool skip_pre_checks);

    // Methods corresponding to the functions
    void performGsDbscan(af::array &X, int D, int minPts, int k, int m, float eps);

    void static preChecks(af::array X, int D, int minPts, int k, int m, float eps);

    void static preProcessing(af::array D);

    void static randomProjections(af::array X, int D, int k, int m, float eps);

    void static constructABMatrices(af::array X, af::array Y, int k, int m);

    af::array static findDistances(af::array X, af::array A, af::array B, float alpha = 1.2);

    int static findDistanceBatchSize(float alpha, int n, int d, int k, int m);

    void constructClusterGraph(af::array distances, float eps, int k, int m);

    af::array GsDBSCAN::assembleAdjacencyList(af::array &distances, af::array &E, af::array &V, af::array &A, af::array &B, float eps, int blockSize=1024);

    af::array constructQueryVectorDegreeArray(af::array distances, float eps);

    af::array GsDBSCAN::processQueryVectorDegreeArray(af::array &E);

    cudaStream_t GsDBSCAN::getAfCudaStream()

    // Destructor if needed
    ~GsDBSCAN() = default;
};

#endif //DBSCANCEOS_GSDBSCAN_H
