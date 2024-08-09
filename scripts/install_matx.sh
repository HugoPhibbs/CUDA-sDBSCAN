cd ~/Documents/MatX
rm -rf build
mkdir build && cd build
cmake ..
make -j
sudo make install