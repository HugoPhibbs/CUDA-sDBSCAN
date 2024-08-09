//
// Created by hphi344 on 10/05/24.
//

#ifndef DBSCANCEOS_GSDBSCAN_H
#define DBSCANCEOS_GSDBSCAN_H

#include "../Header.h"
#include "../Utilities.h"
#include <arrayfire.h>
#include "../../../../../../usr/local/cuda-12.5/targets/x86_64-linux/include/cuda_runtime.h"
#include <af/cuda.h>
#include <arrayfire.h>
#include <matx.h>
#include <Eigen/Dense>

#include "preprocessing.h"
#include "distances.h"
#include "utils.h"
#include "clustering.h"


namespace GsDBSCAN {


    /**
    * Performs the gs dbscan algorithm
    *
    * @param X ArrayFire af::array matrix for the X data points
    * @param D int for number of random vectors to generate
    * @param minPts min number of points as per the DBSCAN algorithm
    * @param k k parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest random vectors to take for ecah data point
    * @param m m parameter for the sDBSCAN algorithm. I.e. the number of closest/furthest dataset vecs for each random vec to take
    * @param eps epsilon parameter for the sDBSCAN algorithm. I.e. the threshold for the distance between the random vec and the dataset vec
    * @param skip_pre_checks boolean flag to skip the pre-checks
    */
    void performGsDbscan() {
//    if (!skip_pre_checks) {
//        preChecks(X, D, minPts, k, m, eps);
//    }
        // Something something ...

        /*
         * Steps:
         *
         * 1. Preprocessing - Perform random projections and create A and B matrices
         * 2. Find distances between query vectors and their candidate vectors
         * 3. Use vectors to create E and V vectors
         * 4. Create the adjacency list
         * 5. Finally, create the cluster graph - can use Ninh's pre-existing clustering method
         *
         *
         */
    }

};

#endif //DBSCANCEOS_GSDBSCAN_H
