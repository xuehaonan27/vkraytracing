set(src_files_base
    Base/Application.cpp
    Base/Application.hpp
    Base/Buffer.cpp
    Base/Buffer.hpp
    Base/BufferUtil.hpp
    Base/CommandBuffers.cpp
    Base/CommandBuffers.hpp
    Base/CommandPool.cpp
    Base/CommandPool.hpp
    Base/DebugUtils.cpp
    Base/DebugUtils.hpp
    Base/DebugUtilsMessenger.cpp
    Base/DebugUtilsMessenger.hpp
    Base/DepthBuffer.cpp
    Base/DepthBuffer.hpp
    Base/DescriptorBinding.hpp
    Base/DescriptorPool.cpp
    Base/DescriptorPool.hpp
    Base/DescriptorSetLayout.cpp
    Base/DescriptorSetLayout.hpp
    Base/DescriptorSetManager.cpp
    Base/DescriptorSetManager.hpp
    Base/DescriptorSets.cpp
    Base/DescriptorSets.hpp
    Base/Device.cpp
    Base/Device.hpp
    Base/DeviceMemory.cpp
    Base/DeviceMemory.hpp
    Base/DeviceProcedures.cpp
    Base/DeviceProcedures.hpp
    Base/Enumerate.hpp
    Base/Fence.cpp
    Base/Fence.hpp
    Base/FrameBuffer.cpp
    Base/FrameBuffer.hpp
    Base/GraphicsPipeline.cpp
    Base/GraphicsPipeline.hpp
    Base/Image.cpp
    Base/Image.hpp	
    Base/ImageMemoryBarrier.hpp	
    Base/ImageView.cpp
    Base/ImageView.hpp	
    Base/Instance.cpp
    Base/Instance.hpp
    Base/PipelineLayout.cpp
    Base/PipelineLayout.hpp
    Base/RenderPass.cpp
    Base/RenderPass.hpp
    Base/Sampler.cpp
    Base/Sampler.hpp
    Base/Scene.cpp
    Base/Scene.hpp
    Base/Semaphore.cpp
    Base/Semaphore.hpp
    Base/ShaderModule.cpp
    Base/ShaderModule.hpp	
    Base/SingleTimeCommands.hpp
    Base/Strings.cpp
    Base/Strings.hpp	
    Base/Surface.cpp
    Base/Surface.hpp	
    Base/SwapChain.cpp
    Base/SwapChain.hpp
    Base/TextureImage.cpp
    Base/TextureImage.hpp
    Base/UniformBuffer.cpp
    Base/UniformBuffer.hpp
    Base/Version.hpp
    Base/Window.cpp
    Base/Window.hpp
    Base/WindowConfig.hpp
)

set(src_files_utils
    Utils/Console.cpp
    Utils/Console.hpp
    Utils/Glm.hpp
    Utils/Macros.cpp
    Utils/Macros.hpp
    Utils/ModelViewController.cpp
    Utils/ModelViewController.hpp
    Utils/Options.cpp
    Utils/Options.hpp
    Utils/SceneList.cpp
    Utils/SceneList.hpp
    Utils/StbImage.cpp
    Utils/StbImage.hpp
    Utils/UserInterface.cpp
    Utils/UserInterface.hpp
    Utils/UserSettings.hpp
)

set(src_files_raytrace
    RayTrace/AccelerationStructure.cpp
    RayTrace/AccelerationStructure.hpp
    RayTrace/Application.cpp
    RayTrace/Application.hpp
    RayTrace/BottomLevelAccelerationStructure.cpp
    RayTrace/BottomLevelAccelerationStructure.hpp
    RayTrace/BottomLevelGeometry.cpp
    RayTrace/BottomLevelGeometry.hpp
    RayTrace/CornellBox.cpp
    RayTrace/CornellBox.hpp
    RayTrace/Material.hpp
    RayTrace/Model.cpp
    RayTrace/Model.hpp
    RayTrace/Procedural.hpp
    RayTrace/RayTracingPipeline.cpp
    RayTrace/RayTracingPipeline.hpp
    RayTrace/RayTracingProperties.cpp
    RayTrace/RayTracingProperties.hpp
    RayTrace/ShaderBindingTable.cpp
    RayTrace/ShaderBindingTable.hpp
    RayTrace/Sphere.hpp
    RayTrace/Texture.cpp
    RayTrace/Texture.hpp
    RayTrace/TopLevelAccelerationStructure.cpp
    RayTrace/TopLevelAccelerationStructure.hpp
    RayTrace/Vertex.hpp
)

set(src_files
    main.cpp
    RayTracer.cpp
    RayTracer.hpp
)

source_group("Base" FILES ${src_files_base})
source_group("Utils" FILES ${src_files_utils})
source_group("RayTrace" FILES ${src_files_raytrace})
source_group("Main" FILES ${src_files})

# Include header file directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils
    ${CMAKE_CURRENT_SOURCE_DIR}/Base
    ${CMAKE_CURRENT_SOURCE_DIR}/RayTrace
)


# Executables
add_executable(
    ${PROJECT_NAME}
    ${src_files}
    ${src_files_base}
    ${src_files_utils}
    ${src_files_raytrace}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${Vulkan_LIBRARIES}
    glfw
    glm::glm
    imgui::imgui
    Boost::program_options
)

# Add dependency on Assets target to ensure shaders are compiled and resources are copied
add_dependencies(${PROJECT_NAME} Assets)
