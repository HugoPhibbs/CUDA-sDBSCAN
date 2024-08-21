cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs LANGUAGES CXX CUDA C)
#enable_language(CUDA)
project(sDbscan)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc") # Somehow CLion needs this here (smh)
#SET(CMAKE_C_COMPILER "/usr/bin/g++")
#add_definitions(-DINDEX_64_BIT)

SET(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.71 REQUIRED NO_MODULE)

# CCCL
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG v2.4.0
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
find_package(CUDAToolkit 12.6 REQUIRED)
#find_package(CUDAToolkit 11.8 REQUIRED)

# ArrayFire
find_package(ArrayFire REQUIRED)

# MatX https://github.com/NVIDIA/MatX
find_package(matx CONFIG REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/lib/eigen-3.4.0
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${gtest_SOURCE_DIR}/include
        ${gtest_SOURCE_DIR}
)

link_directories(
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

add_subdirectory(lib/googletest)

set_source_files_properties(
        test/gsDBSCAN/GsDBSCANTest.cpp
        test/gsDBSCAN/UtilsTest.cpp
        test/gsDBSCAN/ProjectionsTest.cpp
        test/gsDBSCAN/DistancesTest.cpp
        test/gsDBSCAN/ClusteringTest.cpp
        include/gsDBSCAN/projections.h
        include/gsDBSCAN/GsDBSCAN.h
        include/gsDBSCAN/projections.h
        include/gsDBSCAN/clustering.h
        include/gsDBSCAN/utils.h
        src/gs_main.cpp
        test/repro2.cu
        PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME}
#        src/main.cpp
        src/gs_main.cpp
        src/Utilities.cpp
        src/dbscan/sDbscan.cpp
        src/fast_copy.c
        src/fht/fht.c
        include/gsDBSCAN/projections.h
        include/gsDBSCAN/utils.h
        include/gsDBSCAN/distances.h
        include/gsDBSCAN/clustering.h
)

add_executable(run_gs_dbscan_tests
        test/gsDBSCAN/GsDBSCANTest.cpp
        test/TestUtils.cpp
        test/gsDBSCAN/UtilsTest.cpp
        test/gsDBSCAN/ProjectionsTest.cpp
        test/gsDBSCAN/DistancesTest.cpp
        test/gsDBSCAN/ClusteringTest.cpp
)

add_executable(repro2
        test/repro2.cu
)

set_target_properties(repro2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(run_gs_dbscan_tests PRIVATE CCCL::CCCL CUDA::cudart ArrayFire::afcuda matx::matx gtest gtest_main)
target_link_libraries(${PROJECT_NAME} PRIVATE CCCL::CCCL CUDA::cudart ArrayFire::afcuda Eigen3::Eigen matx::matx)

target_precompile_headers(run_gs_dbscan_tests PRIVATE include/json.hpp include/rapidcsv.h)
target_precompile_headers(${PROJECT_NAME} PRIVATE include/json.hpp include/rapidcsv.h)

target_link_libraries(repro2 PRIVATE CCCL::CCCL matx::matx)