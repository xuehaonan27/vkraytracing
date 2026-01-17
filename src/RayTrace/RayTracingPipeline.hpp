#ifndef RAYTRACE_RAYTRACINGPIPELINE_HPP
#define RAYTRACE_RAYTRACINGPIPELINE_HPP

#include <memory>
#include <vector>

#include "Base/DescriptorSetManager.hpp"
#include "Base/DeviceProcedures.hpp"
#include "Base/ImageView.hpp"
#include "Base/PipelineLayout.hpp"
#include "Base/Scene.hpp"
#include "Base/SwapChain.hpp"
#include "Base/UniformBuffer.hpp"
#include "Utils/Macros.hpp"

namespace RayTrace {
class DeviceProcedures;
class TopLevelAccelerationStructure;

class RayTracingPipeline final {
  public:
    VULKAN_NON_COPIABLE(RayTracingPipeline)

    RayTracingPipeline(
        const Base::DeviceProcedures& deviceProcedures,
        const Base::SwapChain& swapChain,
        const TopLevelAccelerationStructure& accelerationStructure,
        const Base::ImageView& accumulationImageView,
        const Base::ImageView& outputImageView,
        const std::vector<Base::UniformBuffer>& uniformBuffers,
        const Base::Scene& scene
    );
    ~RayTracingPipeline();

    uint32_t RayGenShaderIndex() const {
        return rayGenIndex_;
    }

    uint32_t MissShaderIndex() const {
        return missIndex_;
    }

    uint32_t TriangleHitGroupIndex() const {
        return triangleHitGroupIndex_;
    }

    uint32_t ProceduralHitGroupIndex() const {
        return proceduralHitGroupIndex_;
    }

    VkDescriptorSet DescriptorSet(size_t index) const;

    const class Base::PipelineLayout& PipelineLayout() const {
        return *pipelineLayout_;
    }

  private:
    const Base::SwapChain& swapChain_;

    VULKAN_HANDLE(VkPipeline, pipeline_)

    std::unique_ptr<class Base::DescriptorSetManager> descriptorSetManager_;
    std::unique_ptr<class Base::PipelineLayout> pipelineLayout_;

    uint32_t rayGenIndex_;
    uint32_t missIndex_;
    uint32_t triangleHitGroupIndex_;
    uint32_t proceduralHitGroupIndex_;
};

} // namespace RayTrace

#endif // RAYTRACE_RAYTRACINGPIPELINE_HPP
