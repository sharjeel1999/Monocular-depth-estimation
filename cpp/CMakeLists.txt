cmake_minimum_required(VERSION 3.10)
project(torch_practice)

# set(CMAKE_PREFIX_PATH "C:/cpp/opencv")
set(CMAKE_PREFIX_PATH "/home/sharjeel/Desktop/cpp_headers/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find the OpenCV package
# This will search for OpenCV and set the OpenCV_DIR variable to the path where OpenCV is installed
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Include the OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# create a new executable target
add_executable(torch_practice main.cpp
    ./networks/resnet_encoder_pre.cpp
)

# Add the OpenCV libraries to the target
# This is the correct way to link libraries in CMake
target_link_libraries(torch_practice ${OpenCV_LIBS})
target_link_libraries(torch_practice ${TORCH_LIBRARIES})