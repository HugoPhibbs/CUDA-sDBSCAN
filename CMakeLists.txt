cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)

## Configuring FetchContent
include(FetchContent)

FetchContent_Declare(
        ArrayFire
        GIT_REPOSITORY https://github.com/arrayfire/arrayfire.git
        GIT_TAG v3.9.0
)

# Declare Google Test
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)

# Declare Eigen
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)

# Configuring ArrayFire

FetchContent_GetProperties(ArrayFire)
if(NOT ArrayFire_POPULATED)
    FetchContent_Populate(ArrayFire)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)  # Disable examples to speed up build
    set(BUILD_TEST OFF CACHE BOOL "" FORCE)  # Disable tests
    add_subdirectory(${arrayfire_SOURCE_DIR} ${arrayfire_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Include the FetchContent managed ArrayFire directories
include_directories(${arrayfire_BINARY_DIR}/include)
link_directories(${arrayfire_BINARY_DIR}/lib)


## Configuring Eigen

# Populate Eigen
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)  # Disable examples to speed up build
    set(BUILD_TEST OFF CACHE BOOL "" FORCE)  # Disable tests
endif()


## Configuring CUDA

# Set the path to the CUDA toolkit. Replace this with your path if necessary.
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.3")

# Include CUDA
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)


## Configuring Google Test

# Populate Google Test
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


## Including Directories

include_directories(
        ${arrayfire_BINARY_DIR}/include
        ${eigen_SOURCE_DIR}
        ${googletest_SOURCE_DIR}/include
)

link_directories(${arrayfire_BINARY_DIR}/lib)

## Configuring Executables

# Define sources for the main project executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/fast_copy.c
        src/dbscan/sngDBSCAN.cpp
        include/sngDBSCAN.h
        src/dbscan/DBSCAN.cpp
        src/dbscan/sDBSCAN.cpp
        include/sDBSCAN.h
        src/dbscan/uDBSCAN.cpp
        include/uDBSCAN.h
        src/dbscan/gsDBSCAN.cpp
        include/gsDBSCAN.h
        src/dbscan/preprocessing.cpp
        include/preprocessing.h
        src/fht/fht.c
)

# Define the test executable
add_executable(run_gs_dbscan_tests
        test/Test.cpp
        test/gsDBSCANTest.cpp
        src/dbscan/gsDBSCAN.cpp  # Assuming test depends on these source files
        src/fht/fht.c            # Include other dependencies as needed
)


## Linking Libraries

# Link Google Test libraries to the test executable
target_link_libraries(run_gs_dbscan_tests gtest gtest_main)

# Link Arrayfire
target_link_libraries(${PROJECT_NAME} ArrayFire::afcuda)
target_link_directories(run_gs_dbscan_tests ArrayFire::afcuda)
