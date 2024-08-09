# Clean Config and Build

rm -rf build && echo "Removed build directory"
mkdir build && echo "Added build directory"

echo "Configuring build" && cmake . -B build
echo "Building" && cmake --build build