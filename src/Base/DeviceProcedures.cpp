#include "DeviceProcedures.hpp"

#include <string>

#include "Device.hpp"
#include "Utils/Macros.hpp"

namespace Base {

namespace {
    template<class Func>
    Func GetProcedure(const Base::Device& device, const char* const name) {
        const auto func = reinterpret_cast<Func>(vkGetDeviceProcAddr(device.Handle(), name));
        if (func == nullptr) {
            throw std::runtime_error(std::string("failed to get address of '") + name + "'");
        }

        return func;
    }
} // namespace

DeviceProcedures::DeviceProcedures(const Base::Device& device) :
    vkCreateAccelerationStructureKHR(
        GetProcedure<PFN_vkCreateAccelerationStructureKHR>(
            device,
            "vkCreateAccelerationStructureKHR"
        )
    ),
    vkDestroyAccelerationStructureKHR(
        GetProcedure<PFN_vkDestroyAccelerationStructureKHR>(
            device,
            "vkDestroyAccelerationStructureKHR"
        )
    ),
    vkGetAccelerationStructureBuildSizesKHR(
        GetProcedure<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            device,
            "vkGetAccelerationStructureBuildSizesKHR"
        )
    ),
    vkCmdBuildAccelerationStructuresKHR(
        GetProcedure<PFN_vkCmdBuildAccelerationStructuresKHR>(
            device,
            "vkCmdBuildAccelerationStructuresKHR"
        )
    ),
    vkCmdCopyAccelerationStructureKHR(
        GetProcedure<PFN_vkCmdCopyAccelerationStructureKHR>(
            device,
            "vkCmdCopyAccelerationStructureKHR"
        )
    ),
    vkCmdTraceRaysKHR(GetProcedure<PFN_vkCmdTraceRaysKHR>(device, "vkCmdTraceRaysKHR")),
    vkCreateRayTracingPipelinesKHR(
        GetProcedure<PFN_vkCreateRayTracingPipelinesKHR>(device, "vkCreateRayTracingPipelinesKHR")
    ),
    vkGetRayTracingShaderGroupHandlesKHR(
        GetProcedure<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            device,
            "vkGetRayTracingShaderGroupHandlesKHR"
        )
    ),
    vkGetAccelerationStructureDeviceAddressKHR(
        GetProcedure<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            device,
            "vkGetAccelerationStructureDeviceAddressKHR"
        )
    ),
    vkCmdWriteAccelerationStructuresPropertiesKHR(
        GetProcedure<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(
            device,
            "vkCmdWriteAccelerationStructuresPropertiesKHR"
        )
    ),
    device_(device) {}

DeviceProcedures::~DeviceProcedures() {}

} // namespace Base
