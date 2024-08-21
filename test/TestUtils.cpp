//

// Created by hphi344 on 30/06/24.
//

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

    af::array createMockMnistDataset(int n, int d) {
        // Create random integers in the range [0, 255]
        af::array X = af::randu(n, d, u16) * 255;

        // Convert to float and scale to [0, 1] range
        X = X.as(f16) / 255.0f;

        return X;
    }

    // Function to create mock A and B matrices
    std::pair<af::array, af::array> createMockABMatrices(int n, int k, int m, int D) {
        af::array A = af::randu(n, 2 * k, f32) * (2 * (D - 1));
        A = A.as(u32);

        af::array B = af::randu(2 * D, m, f32) * (n - 1);
        B = B.as(u32);

        return std::make_pair(A, B);
    }

    af::array createMockRandomVectorsSet(int D, int d) {
        af::array Y = af::randu(d, D, f32);

        Y = Y.as(f16);

        return Y;
    }

    af::array createMockDistances(int n, int D) {
        af::array distances = af::randu(n, D, f32);

//        distances = distances.as(f16); // sort doesn't work with f16

        return distances;
    }

    bool arraysEqual(const af::array &a, const af::array &b) {
        return af::allTrue<bool>(a == b);
    }

    bool arraysApproxEqual(const af::array &a, const af::array &b, double eps) {
        return af::allTrue<bool>(af::abs(a - b) <= eps);
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

    af::array csvToArray(const std::string &filename) {
        std::vector<std::vector<float>> data = readCSV(filename);
        int n = data.size();
        int m = data.at(0).size();

        af::array array(n, m, f32); // Create array with matching dimensions and data type (f32 for float)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                array(i, j) = data[i][j];
            }
        }

        return array;
    }
}

