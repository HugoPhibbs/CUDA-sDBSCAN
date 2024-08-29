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

    auto args = GsDBSCAN::run_utils::parseArgs(argc, argv);

    std::cout << "Args: " << args.dump(4) << std::endl;

    auto [clusterLabels, numClusters, times] = GsDBSCAN::run_utils::main_helper(
            args["datasetFilename"],
            args["n"],
            args["d"],
            args["D"],
            args["minPts"],
            args["k"],
            args["m"],
            args["eps"],
            args["alpha"],
            args["distancesBatchSize"],
            args["distanceMetric"],
            args["clusterBlockSize"],
            args["clusterOnCpu"],
            args["projectionsMethod"],
            args["needToNormalize"]
    );

    GsDBSCAN::run_utils::writeResults(args, times, clusterLabels, numClusters);

    std::cout << "Times: " << times.dump(4) << std::endl;
    std::cout << "NumClusters: " << numClusters << std::endl;

    return 0;
}