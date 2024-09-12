//
// Created by hphi344 on 19/08/24.
//

#include <string>
#include "../include/pch.h"
#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "../include/gsDBSCAN/run_utils.h"

using json = nlohmann::json;


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Error: No arguments provided" << std::endl;
        return 1; // No arguments
    }

    std::cout << "Running GsDBSCAN-CPP" << std::endl;

    auto params = GsDBSCAN::parseArgs(argc, argv);

    std::cout << "Params: " << params.toString() << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(params);

    GsDBSCAN::run_utils::writeResults(params, times, clusterLabels, numClusters);

    std::cout << "Times: " << times.dump(4) << std::endl;
    std::cout << "NumClusters: " << numClusters << std::endl;

    return 0;
}