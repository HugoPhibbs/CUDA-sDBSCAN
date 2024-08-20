//
// Created by hphi344 on 19/08/24.
//

#include <string>
#include <unordered_map>
#include <vector>

#include "../include/json.hpp"
#include "../include/gsDBSCAN/GsDBSCAN.h"
#include "cuda_runtime.h"
#include "../include/gsDBSCAN/utils.h"
#include <arrayfire.h>

using json = nlohmann::json;


json cleanArgs(json args) {
    json newArgs;

    newArgs["datasetFilename"] = args["--datasetFilename"];
    newArgs["outFile"] = args["--outFile"];

    newArgs["n"] = std::stoi((string) args["--n"]);
    newArgs["d"] = std::stoi((string) args["--d"]);

    newArgs["D"] = std::stoi((string) args["--D"]);
    newArgs["minPts"] = std::stoi((string) args["--minPts"]);
    newArgs["k"] = std::stoi((string) args["--k"]);
    newArgs["m"] = std::stoi((string) args["--m"]);
    newArgs["eps"] = std::stof((string) args["--eps"]);

    newArgs["distanceMetric"] = args["--distanceMetric"];

    newArgs["clusterBlockSize"] = std::stoi((string) args["--clusterBlockSize"]);
    newArgs["timeIt"] = std::stoi((string) args["--timeIt"]);

    return newArgs;
}

json parseArgs(int argc, char *argv[]) {
    json args;

    // Parse arguments
    for (int i = 1; i < argc; i += 2) {
        std::string key = argv[i];
        if (i + 1 < argc) {
            std::string value = argv[i + 1];
            args[key] = value;
        } else {
            throw std::runtime_error("Error: Missing value for argument: " + key);
        }
    }

    return cleanArgs(args);
}

float *loadDatasetToDevice(std::string filename, int n, int d) {
    std::vector<float> X_vec = GsDBSCAN::utils::loadCsvColumnToVector<float>(filename, 0);
    float *X_h = X_vec.data();
    float *X_d = GsDBSCAN::utils::copyHostToDevice(X_h, n * d);
    return X_d;
}

void writeResults(json args, json times, int* clusterLabels, int* typeLabels) {
    std::ofstream file(args["outFile"]);
    json combined;
    combined["args"] = args;
    combined["times"] = times;

    std::vector<int> clusterLabelsVec(clusterLabels, clusterLabels + (size_t) args["n"]);
    std::vector<int> typeLabelsVec(typeLabels, typeLabels + (size_t) args["n"]);
    combined["clusterLabels"] = clusterLabelsVec;
    combined["typeLabels"] = typeLabelsVec;

    if (file.is_open()) {
        file << combined.dump(4);
        file.close();
    } else {
        throw std::runtime_error("Error: Unable to open file: " + (string) args["outFile"]);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Error: No arguments provided" << std::endl;
        return 1; // No arguments
    }

    auto args = parseArgs(argc, argv);

    std::cout << "Args: " << args.dump(4) << std::endl;

    float *X_d = loadDatasetToDevice(args["datasetFilename"], args["n"], args["d"]);

    auto [clusterLabels, typeLabels, times] = GsDBSCAN::performGsDbscan(
            X_d,
            args["n"],
            args["d"],
            args["D"],
            args["minPts"],
            args["k"],
            args["m"],
            args["eps"],
            args["alpha"],
            args["distanceMetric"],
            args["--clusterBlockSize"],
            args["--timeIt"] // 0: no timing, 1: timing
    );

    cudaFree(X_d);

    writeResults(args, times, clusterLabels, typeLabels);

    return 0;
}