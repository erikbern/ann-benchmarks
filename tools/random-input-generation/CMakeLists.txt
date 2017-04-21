# CMake file for random input generator project

project(randominput)

cmake_minimum_required(VERSION 2.8)

# disallow in-source builds

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(SEND_ERROR "In-source builds are not allowed.")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

# Enable warnings

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")


# Use TCLAP

include_directories( BEFORE tclap-1.2.1/include )
include_directories (src/)

# C++14

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

# PThread

if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")
endif("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")

# Debug symbols

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Save Assembler Code

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -S")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -funroll-loops")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
#string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# descend into source
add_subdirectory(src)
