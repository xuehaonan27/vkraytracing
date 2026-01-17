#ifndef RAYTRACE_BOTTOM_LEVEL_ACCELERATION_STRUCTURE_HPP
#define RAYTRACE_BOTTOM_LEVEL_ACCELERATION_STRUCTURE_HPP

#include "AccelerationStructure.hpp"
#include "BottomLevelGeometry.hpp"

namespace RayTrace {
class Procedural;

class BottomLevelAccelerationStructure final: public AccelerationStructure {
  public:
    BottomLevelAccelerationStructure(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(BottomLevelAccelerationStructure&&) = delete;

    BottomLevelAccelerationStructure(
        const class Base::DeviceProcedures& deviceProcedures,
        const class RayTracingProperties& rayTracingProperties,
        const BottomLevelGeometry& geometries
    );
    BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept;
    ~BottomLevelAccelerationStructure();

    void Generate(
        VkCommandBuffer commandBuffer,
        Base::Buffer& scratchBuffer,
        VkDeviceSize scratchOffset,
        Base::Buffer& resultBuffer,
        VkDeviceSize resultOffset
    );

  private:
    BottomLevelGeometry geometries_;
};

} // namespace RayTrace

#endif // RAYTRACE_BOTTOM_LEVEL_ACCELERATION_STRUCTURE_HPP
