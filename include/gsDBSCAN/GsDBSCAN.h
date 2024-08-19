//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include "../Header.h"
#include "../Utilities.h"
#include <arrayfire.h>
#include "cuda_runtime.h"
#include <af/cuda.h>
#include <arrayfire.h>
#include <matx.h>
#include <Eigen/Dense>
#include <tuple>

#include "projections.h"
#include "distances.h"
#include "utils.h"
#include "clustering.h"

namespace GsDBSCAN {

    /**
    * Performs the gs dbscan algorithm
    *
    * @param X array storing the dataset. Should be in ROW major order. Stored on the GPU
    * @param n number of entries in the X dataset
    * @param d dimension of the X dataset
    * @param D int for number of random vectors to generate
    * @param minPts min number of points as per the DBSCAN algorithm
    * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
    * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
    * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
    * @param alpha float to tune the batch size when calculating distances
    * @param distanceMetric string for the distance metric to use. Options are "L1", "L2" or "COSINE"
    * @return a tuple containing:
    *  An integer array of size n containing the cluster labels for each point in the X dataset
    *  An integer array of size n containing the type labels for each point in the X dataset - e.g. Noise, Core, Border // TODO decide on how this will work?
    */
    inline std::tuple<int*, int*>  performGsDbscan(float *X, int n, int d, int D, int minPts, int k, int m, float eps, float alpha=1.2, std::string distanceMetric="L2") {
        auto X_col_major = utils::colMajorToRowMajorMat(X, n, d);
        auto X_af = af::array(n, d, X_col_major);
        auto projections = projections::performProjections(X_af, D);

        auto [A_af, B_af] = projections::constructABMatricesAF(projections, k, m);

        auto A_t = utils::afMatToMatXTensor<int, int>(A_af,
                                                      matx::MATX_DEVICE_MEMORY); // TODO use MANAGED or DEVICE memory?
        auto B_t = utils::afMatToMatXTensor<int, int>(B_af,
                                                      matx::MATX_DEVICE_MEMORY); // TODO use MANAGED or DEVICE memory?
        auto X_t = utils::afMatToMatXTensor<float, float>(X_af, matx::MATX_DEVICE_MEMORY);

        matx::tensor_t<float, 2> distances = distances::findDistancesMatX(X_t, A_t, B_t, alpha, -1, distanceMetric,
                                                                          matx::MATX_DEVICE_MEMORY);
        int *degArray_d = clustering::constructQueryVectorDegreeArrayMatx(distances, eps);
        int *startIdxArray_d = clustering::processQueryVectorDegreeArrayThrust(degArray_d, n);

        auto [adjacencyList_d, adjacencyList_size] = clustering::constructAdjacencyList(distances.Data(), degArray_d,
                                                                                        startIdxArray_d, A_t.Data(),
                                                                                        B_t.Data(), n, k, m, eps);

        return clustering::formClusters(adjacencyList_d, startIdxArray_d, degArray_d, n, minPts);
    }


};

#endif //DBSCANCEOS_GSDBSCAN_H
