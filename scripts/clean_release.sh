cd ~/Documents/GS-DBSCAN-CPP
rm -rf build
mkdir build
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cd build
cmake --build . --config Release -- -j50

echo "Build complete"