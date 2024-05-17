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
    void performGsDbscan();

    void preChecks(af::array X, af::array D, int minPts, int k, int m, float eps);

    void preProcessing(af::array D);

    void randomProjections(af::array X, af::array D, int k, int m, float eps);

    void constructABMatrices(af::array X, af::array D, int k, int m);

    void findDistances(af::array X, af::array A, af::array B, float alpha = 1.2);

    int findDistanceBatchSize(float alpha);

    void constructClusterGraph(af::array distances, float eps, int k, int m);

    void assembleAdjacencyList(af::array distances, int E, int V, af::array A, af::array B, float eps, int blockSize = 1024);

    // Destructor if needed
    ~GsDBSCAN() = default;
};

#endif //DBSCANCEOS_GSDBSCAN_H
