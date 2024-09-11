//
// Created by hphi344 on 12/09/24.
//

#include <string>

#ifndef SDBSCAN_GSDBSCAN_PARAMS_H
#define SDBSCAN_GSDBSCAN_PARAMS_H

namespace GsDBSCAN {

    class GsDBSCAN_Params {
    private:

        float adjustEps(float _eps) {
            if (this->distanceMetric == "COSINE") {
                return 1 - _eps; // We use cosine similarity, thus we need to convert the eps to a cosine distance.
            }
            return _eps;
        }


    public:
        int n;
        int d;
        int D;
        int minPts;
        int k;
        int m;
        float eps;
        float alpha;
        int distancesBatchSize;
        std::string distanceMetric;
        int clusterBlockSize;
        bool timeIt;
        bool clusterOnCpu;
        bool needToNormalise;
        int fourierEmbedDim;
        float sigmaEmbed;
        int ABatchSize;
        int BBatchSize;
        int miniBatchSize;

        GsDBSCAN_Params(int n, int d, int D, int minPts, int k, int m, float eps, float alpha, int distancesBatchSize,
                        const std::string &distanceMetric, int clusterBlockSize, bool timeIt, bool clusterOnCpu,
                        bool needToNormalise, int fourierEmbedDim, float sigmaEmbed, int ABatchSize = 10000,
                        int BBatchSize = 128, int miniBatchSize = 10000) {

            this->n = n;
            this->d = d;
            this->D = D;
            this->minPts = minPts;
            this->k = k;
            this->m = m;
            this->alpha = alpha;
            this->distancesBatchSize = distancesBatchSize;
            this->distanceMetric = distanceMetric;
            this->eps = adjustEps(eps);
            this->clusterBlockSize = clusterBlockSize;
            this->timeIt = timeIt;
            this->clusterOnCpu = clusterOnCpu;
            this->needToNormalise = needToNormalise;
            this->fourierEmbedDim = fourierEmbedDim;
            this->sigmaEmbed = sigmaEmbed;
            this->ABatchSize = ABatchSize;
            this->BBatchSize = BBatchSize;
            this->miniBatchSize = miniBatchSize;
        }


    };
}

#endif //SDBSCAN_GSDBSCAN_PARAMS_H
