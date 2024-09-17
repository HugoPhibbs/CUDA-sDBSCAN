//
// Created by hphi344 on 21/08/24.
//

#ifndef SDBSCAN_RUN_UTILS_H
#define SDBSCAN_RUN_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "../pch.h"
#include "algo_utils.h"
#include "GsDBSCAN.h"
#include "GsDBSCAN_Params.h"

using json = nlohmann::json;


namespace GsDBSCAN::run_utils {

    template<typename T>
    inline std::vector<T> loadBinFileToVector(const std::string &filePath) {
        // Use this instead of the loadBinDatasetToDevice function, I'm pre sure thats broken

        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + filePath);
        }

        // Get the size of the file
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read the file into a vector
        std::vector<T> data(fileSize / sizeof(T));
        file.read(reinterpret_cast<char *>(data.data()), fileSize);

        file.close();

        return data;
    }

    template<typename T>
    inline std::vector<T> loadCsvColumnToVector(const std::string &filePath, size_t columnIndex = 1) {
        rapidcsv::Document csvDoc(filePath);
        return csvDoc.GetColumn<T>(columnIndex);
    }

    inline void
    writeResults(GsDBSCAN_Params params, nlohmann::ordered_json &times, int *clusterLabels, int numClusters) {
        std::ofstream file(params.outputFilename);
        json combined;
        combined["args"] = params.toString();
        combined["times"] = times;

        std::vector<int> clusterLabelsVec(clusterLabels, clusterLabels + (size_t) params.n);
        combined["numClusters"] = numClusters;
        combined["clusterLabels"] = clusterLabelsVec;

        json result = json::array(); // Array of JSON objects, so Pandas can read it
        result.push_back(combined);

        if (file.is_open()) {
            file << result.dump(4);
            file.close();
        } else {
            throw std::runtime_error("Error: Unable to open file: " + (std::string) params.outputFilename);
        }

        delete[] clusterLabels;
    }


    inline std::tuple<int *, int, nlohmann::ordered_json>
    main_helper(GsDBSCAN_Params &params) {
        auto X = loadBinFileToVector<float>(params.dataFilename);
        auto X_h = X.data();

        auto [clusterLabels, numClusters, times] = performGsDbscan(X_h, params);

//        cudaFree(X_h);

        return std::tie(clusterLabels, numClusters, times);
    }
}


#endif //SDBSCAN_RUN_UTILS_H
