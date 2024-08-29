cd ~/Documents/GS-DBSCAN-CPP
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target GS-DBSCAN --config Release -- -j50

echo "Build complete"