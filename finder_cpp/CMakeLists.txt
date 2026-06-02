cmake_minimum_required(VERSION 3.15)
project(finder_cpp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

# Option to build Python bindings

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Source files
set(FINDER_SOURCES
    src/dbscan.cpp
    src/auxiliary_functions.cpp
    src/dbscan_loop.cpp
    src/similarity_score.cpp
    src/finder.cpp
)

# Create finder library
add_library(finderlib STATIC ${FINDER_SOURCES})

# Enable Position Independent Code for use in shared libraries (Python module)
set_target_properties(finderlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(finderlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Create main finder executable (CLI tool)
add_executable(finder
    src/main.cpp
)

target_link_libraries(finder PRIVATE finderlib)


# Installation for CLI tool and library
install(TARGETS finderlib finder
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY docs/ DESTINATION share/doc/finder)
