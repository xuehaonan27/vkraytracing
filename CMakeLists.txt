cmake_minimum_required(VERSION 3.20)
project (vkraytracing VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Vulkan REQUIRED)
if (Vulkan_FOUND)
    message(STATUS "Found Vulkan: ${Vulkan_INCLUDE_DIRS}")
endif()

find_package(glfw3 REQUIRED)
if (glfw3_FOUND)
    message(STATUS "Found GLFW3: ${glfw3_INCLUDE_DIRS}")
endif()

find_package(glm REQUIRED)
if (glm_FOUND)
    message(STATUS "Found GLM: ${glm_INCLUDE_DIRS}")
endif()

# Include header file directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${Vulkan_INCLUDE_DIRS}
    ${glfw3_INCLUDE_DIRS}
    ${glm_INCLUDE_DIRS}
)


add_executable(
    ${PROJECT_NAME}
    main.cpp
    # src/foo.cpp ...
)

target_link_libraries(
    ${PROJECT_NAME}
    ${Vulkan_LIBRARIES}
    glfw
    glm::glm
)

# Copy textures and models
