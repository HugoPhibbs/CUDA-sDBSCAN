cd ~/Documents/GS-DBSCAN-CPP
rm -rf build-release
mkdir build-release
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j50

echo "Build complete"