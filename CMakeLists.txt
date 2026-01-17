cmake_minimum_required(VERSION 3.20)
project (vkraytracing VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Dependency checking
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options) 

# Include header file directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${glfw3_INCLUDE_DIRS}
    ${glm_INCLUDE_DIRS}
)

# Used to compiler shaders
find_program(SHADERC_EXECUTABLE glslangValidator HINTS $ENV{VULKAN_SDK}/Bin)
if(NOT SHADERC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found!")
endif()

add_subdirectory(assets)
add_subdirectory(src)
