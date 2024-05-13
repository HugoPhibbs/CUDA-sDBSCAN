cmake_minimum_required(VERSION 3.27)
project(DbscanCEOs)

set(CMAKE_CXX_STANDARD 17)

# Add Google Test subdirectory
add_subdirectory(googletest)

# Include Google Test in the include path
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Define sources for the main project executable
add_executable(yourProj
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
        src/fht/fht.c
)

# Define the test executable
add_executable(run_gs_dbscan_tests
        test/Test.cpp
        test/gsDBSCANTest.cpp
        src/dbscan/gsDBSCAN.cpp  # Assuming test depends on these source files
        src/fht/fht.c            # Include other dependencies as needed
)

# Link Google Test libraries to the test executable
target_link_libraries(run_gs_dbscan_tests gtest gtest_main)
