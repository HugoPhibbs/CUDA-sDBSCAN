//
// Created by hphi344 on 19/08/24.
//

#include <string>
#include "../include/json.hpp"
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

    auto [clusterLabels, typeLabels, times] = GsDBSCAN::run_utils::main_helper(
            args["datasetFilename"],
            args["n"],
            args["d"],
            args["D"],
            args["minPts"],
            args["k"],
            args["m"],
            args["eps"],
            args["alpha"],
            args["distanceMetric"],
            args["clusterBlockSize"]
    );

    GsDBSCAN::run_utils::writeResults(args, times, clusterLabels, typeLabels);

    return 0;
}