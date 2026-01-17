#ifndef RAYTRACE_TOPOLEVEL_ACCELERATION_STRUCTURE_HPP
#define RAYTRACE_TOPOLEVEL_ACCELERATION_STRUCTURE_HPP

#include <vector>

#include "AccelerationStructure.hpp"
#include "Utils/Glm.hpp"

namespace RayTrace {
class BottomLevelAccelerationStructure;

class TopLevelAccelerationStructure final: public AccelerationStructure {
  public:
    TopLevelAccelerationStructure(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(TopLevelAccelerationStructure&&) = delete;

    TopLevelAccelerationStructure(
        const class Base::DeviceProcedures& deviceProcedures,
        const class RayTracingProperties& rayTracingProperties,
        VkDeviceAddress instanceAddress,
        uint32_t instancesCount
    );
    TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept;
    virtual ~TopLevelAccelerationStructure();

    void Generate(
        VkCommandBuffer commandBuffer,
        Base::Buffer& scratchBuffer,
        VkDeviceSize scratchOffset,
        Base::Buffer& resultBuffer,
        VkDeviceSize resultOffset
    );

    static VkAccelerationStructureInstanceKHR CreateInstance(
        const BottomLevelAccelerationStructure& bottomLevelAs,
        const glm::mat4& transform,
        uint32_t instanceId,
        uint32_t hitGroupId
    );

  private:
    uint32_t instancesCount_;
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk_ {};
    VkAccelerationStructureGeometryKHR topASGeometry_ {};
};

} // namespace RayTrace

#endif // RAYTRACE_TOPOLEVEL_ACCELERATION_STRUCTURE_HPP
