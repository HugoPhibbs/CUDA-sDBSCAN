cd ~/Documents/GS-DBSCAN-CPP
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cd build
cmake --build . --config Release -- -j50

echo "Build complete"