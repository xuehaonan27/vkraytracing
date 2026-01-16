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

# Used to compiler shaders
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/Bin)
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found!")
endif()

function (add_slang_shader_target TARGET)
    cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
    set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set(SHADER_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)

    add_custom_command(
        OUTPUT ${SHADER_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    )

    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        set(OUTPUT_FILE ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
        
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${SLANGC_EXECUTABLE}
                ${SHADERS_DIR}/${SHADER_SOURCE}
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                ${ENTRY_POINTS}
                -o ${OUTPUT_FILE}
            DEPENDS ${SHADERS_DIR}/${SHADER_SOURCE} ${SHADER_OUTPUT_DIR}
            COMMENT "Compiling Slang shader: ${SHADER_SOURCE}"
            VERBATIM
        )
        list(APPEND OUTPUT_FILES ${OUTPUT_FILE})
    endforeach()

   add_custom_target(${TARGET} DEPENDS ${OUTPUT_FILES})
endfunction()

# Executables
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

add_slang_shader_target(shader_target SOURCES shader.slang)
add_dependencies(${PROJECT_NAME} shader_target)

# Copy textures and models
set(RESOURCE_DIRS
    textures
    models
)

foreach(RESOURCE_DIR ${RESOURCE_DIRS})
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR})
    set(DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${RESOURCE_DIR})
    
    if(EXISTS ${SOURCE_DIR})
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${DEST_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${SOURCE_DIR} ${DEST_DIR}
            COMMENT "Copying ${RESOURCE_DIR} to output"
        )
        message(STATUS "Added resource directory: ${RESOURCE_DIR}")
    else()
        message(STATUS "Resource directory not found: ${RESOURCE_DIR}")
    endif()
endforeach()

# set(IMPORTANT_FILES
#     README.md
#     LICENSE
#     config.json
# )

# foreach(FILE ${IMPORTANT_FILES})
#     set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${FILE})
#     set(DEST_FILE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FILE})
    
#     if(EXISTS ${SOURCE_FILE})
#         add_custom_command(
#             TARGET ${PROJECT_NAME} POST_BUILD
#             COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE} ${DEST_FILE}
#             COMMENT "Copying ${FILE}"
#         )
#     endif()
# endforeach()
