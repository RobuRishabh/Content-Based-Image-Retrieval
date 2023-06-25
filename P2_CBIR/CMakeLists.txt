cmake_minimum_required(VERSION 3.16.3)

# enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set project name
project( P2_CBIR )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


#add all the executable files
add_executable(featureExtractor src/featureExtractor.cpp src/features.cpp src/distanceMetrics.cpp src/csv_util.cpp)
add_executable(imageRetriever src/imageRetriever.cpp src/features.cpp src/distanceMetrics.cpp src/csv_util.cpp)

#linking libraries
target_link_libraries(featureExtractor ${OpenCV_LIBS})
target_link_libraries(imageRetriever ${OpenCV_LIBS})

