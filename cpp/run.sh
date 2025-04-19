#!/bin/bash

# Navigate to the build directory (if you're not already there)

# mkdir build
cd build
# cmake .. # only once at the beginning


# Build the project using make
make

# Run the executable (replace YourExecutableName with your actual executable name)
./torch_practice

# Optionally, you can navigate back to the root directory
cd ..