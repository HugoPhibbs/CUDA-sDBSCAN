//

// Created by hphi344 on 30/06/24.
//

#include "../include/pch.h"
#include "../include/TestUtils.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

namespace testUtils {
    Time timeNow() {
        return std::chrono::high_resolution_clock::now();
    }


    int duration(Time start, Time stop) {
        return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }

    int durationSecs(Time start, Time stop) {
        return std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
    }


    void printDuration(Time start, Time stop) {
        std::cout << "Duration: " << duration(start, stop) << " microseconds" << std::endl;
    }

    void printDurationSinceStart(Time start, const std::string& msg) {
        if (!msg.empty()) {
            std::cout << msg << ": ";
        }
        std::cout << "Duration: " << duration(start, timeNow()) << " microseconds" << std::endl;
    }

    void printDurationSinceStartSecs(Time start, const std::string& msg) {
        if (!msg.empty()) {
            std::cout << msg << " ";
        }
        std::cout << "Duration: " << durationSecs(start, timeNow()) << " seconds" << std::endl;
    }

    /**
     * Reads a CSV file and returns a vector of vectors of floats
     *
     * Used for testing, can use python as a baseline to generate test data
     *
     * @param filename
     * @return
     * @return
     */
    std::vector<std::vector<float>> readCSV(const std::string &filename) {
        std::ifstream file(filename);
        std::vector<std::vector<float>> data;
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<float> row;

            while (std::getline(lineStream, cell, ',')) {
                row.push_back(std::stof(cell));
            }

            data.push_back(row);
        }

        return data;
    }
}

