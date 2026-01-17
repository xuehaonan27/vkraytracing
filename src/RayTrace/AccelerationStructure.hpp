#ifndef RAYTRACE_ACCELERATION_STRUCTURE_HPP
#define RAYTRACE_ACCELERATION_STRUCTURE_HPP

#include "Base/Buffer.hpp"
#include "Base/Device.hpp"
#include "Base/DeviceProcedures.hpp"
#include "Utils/Macros.hpp"

namespace RayTrace {
class AccelerationStructure {
  public:
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(AccelerationStructure&&) = delete;

    AccelerationStructure(AccelerationStructure&& other) noexcept;
    virtual ~AccelerationStructure();

    const class Base::Device& Device() const {
        return device_;
    }

    const class Base::DeviceProcedures& DeviceProcedures() const {
        return deviceProcedures_;
    }

    const VkAccelerationStructureBuildSizesInfoKHR BuildSizes() const {
        return buildSizesInfo_;
    }

    static void MemoryBarrier(VkCommandBuffer commandBuffer);

  protected:
    explicit AccelerationStructure(
        const class Base::DeviceProcedures& deviceProcedures,
        const class RayTracingProperties& rayTracingProperties
    );

    VkAccelerationStructureBuildSizesInfoKHR
    GetBuildSizes(const uint32_t* pMaxPrimitiveCounts) const;
    void CreateAccelerationStructure(Base::Buffer& resultBuffer, VkDeviceSize resultOffset);

    const class Base::DeviceProcedures& deviceProcedures_;
    const VkBuildAccelerationStructureFlagsKHR flags_;

    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo_ {};
    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo_ {};

  private:
    const class Base::Device& device_;
    const class RayTracingProperties& rayTracingProperties_;

    VULKAN_HANDLE(VkAccelerationStructureKHR, accelerationStructure_)
};

} // namespace RayTrace

#endif // RAYTRACE_ACCELERATION_STRUCTURE_HPP
