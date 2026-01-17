#ifndef RAYTRACE_APPLICATION_HPP
#define RAYTRACE_APPLICATION_HPP

#include "Base/Application.hpp"
#include "RayTracingProperties.hpp"

namespace Base {
class CommandBuffers;
class Buffer;
class DeviceMemory;
class Image;
class ImageView;
} // namespace Base

namespace RayTrace {
class Application: public Base::Application {
  public:
    VULKAN_NON_COPIABLE(Application);

  protected:
    Application(
        const Base::WindowConfig& windowConfig,
        VkPresentModeKHR presentMode,
        bool enableValidationLayers
    );
    ~Application();

    void SetPhysicalDevice(
        VkPhysicalDevice physicalDevice,
        std::vector<const char*>& requiredExtensions,
        VkPhysicalDeviceFeatures& deviceFeatures,
        void* nextDeviceFeatures
    ) override;

    void OnDeviceSet() override;
    void CreateAccelerationStructures();
    void DeleteAccelerationStructures();
    void CreateSwapChain() override;
    void DeleteSwapChain() override;
    void Render(VkCommandBuffer commandBuffer, size_t currentFrame, uint32_t imageIndex) override;

  private:
    void CreateBottomLevelStructures(VkCommandBuffer commandBuffer);
    void CreateTopLevelStructures(VkCommandBuffer commandBuffer);
    void CreateOutputImage();

    std::unique_ptr<class Base::DeviceProcedures> deviceProcedures_;
    std::unique_ptr<class RayTracingProperties> rayTracingProperties_;

    std::vector<class BottomLevelAccelerationStructure> bottomAs_;
    std::unique_ptr<Base::Buffer> bottomBuffer_;
    std::unique_ptr<Base::DeviceMemory> bottomBufferMemory_;
    std::unique_ptr<Base::Buffer> bottomScratchBuffer_;
    std::unique_ptr<Base::DeviceMemory> bottomScratchBufferMemory_;
    std::vector<class TopLevelAccelerationStructure> topAs_;
    std::unique_ptr<Base::Buffer> topBuffer_;
    std::unique_ptr<Base::DeviceMemory> topBufferMemory_;
    std::unique_ptr<Base::Buffer> topScratchBuffer_;
    std::unique_ptr<Base::DeviceMemory> topScratchBufferMemory_;
    std::unique_ptr<Base::Buffer> instancesBuffer_;
    std::unique_ptr<Base::DeviceMemory> instancesBufferMemory_;

    std::unique_ptr<Base::Image> accumulationImage_;
    std::unique_ptr<Base::DeviceMemory> accumulationImageMemory_;
    std::unique_ptr<Base::ImageView> accumulationImageView_;

    std::unique_ptr<Base::Image> outputImage_;
    std::unique_ptr<Base::DeviceMemory> outputImageMemory_;
    std::unique_ptr<Base::ImageView> outputImageView_;

    std::unique_ptr<class RayTracingPipeline> rayTracingPipeline_;
    std::unique_ptr<class ShaderBindingTable> shaderBindingTable_;
};

} // namespace RayTrace

#endif // RAYTRACE_APPLICATION_HPP
