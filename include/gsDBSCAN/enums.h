//
// Created by hphi344 on 30/08/24.
//

#ifndef SDBSCAN_ENUMS_H
#define SDBSCAN_ENUMS_H

#include <string>
#include <stdexcept>

#pragma once

namespace GsDBSCAN {
    enum class DistanceMetric {
        L1,
        L2,
        COSINE
    };

    std::string distanceMetricToString(DistanceMetric distanceMetric) {
        switch (distanceMetric) {
            case DistanceMetric::L1:
                return "L1";
            case DistanceMetric::L2:
                return "L2";
            case DistanceMetric::COSINE:
                return "COSINE";
            default:
                throw std::runtime_error("Error: Unknown distance metric");
        }
    }

    DistanceMetric stringToDistanceMetric(const std::string &distanceMetric) {
        if (distanceMetric == "L1") {
            return DistanceMetric::L1;
        } else if (distanceMetric == "L2") {
            return DistanceMetric::L2;
        } else if (distanceMetric == "COSINE") {
            return DistanceMetric::COSINE;
        } else {
            throw std::runtime_error("Error: Unknown distance metric");
        }
    }
}


#endif //SDBSCAN_ENUMS_H
