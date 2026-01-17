#ifndef BASE_DEVICE_HPP
#define BASE_DEVICE_HPP

#include <vector>

#include "DebugUtils.hpp"
#include "Utils/Macros.hpp"

namespace Base {
class Surface;

class Device final {
  public:
    VULKAN_NON_COPIABLE(Device)

    Device(
        VkPhysicalDevice physicalDevice,
        const Surface& surface,
        const std::vector<const char*>& requiredExtensionsconst,
        const VkPhysicalDeviceFeatures& deviceFeatures,
        const void* nextDeviceFeatures
    );

    ~Device();

    VkPhysicalDevice PhysicalDevice() const {
        return physicalDevice_;
    }

    const class Surface& Surface() const {
        return surface_;
    }

    const class DebugUtils& DebugUtils() const {
        return debugUtils_;
    }

    uint32_t GraphicsFamilyIndex() const {
        return graphicsFamilyIndex_;
    }

    uint32_t PresentFamilyIndex() const {
        return presentFamilyIndex_;
    }

    VkQueue GraphicsQueue() const {
        return graphicsQueue_;
    }

    VkQueue PresentQueue() const {
        return presentQueue_;
    }

    void WaitIdle() const;

  private:
    void CheckRequiredExtensions(
        VkPhysicalDevice physicalDevice,
        const std::vector<const char*>& requiredExtensions
    ) const;

    const VkPhysicalDevice physicalDevice_;
    const class Surface& surface_;

    VULKAN_HANDLE(VkDevice, device_)

    class DebugUtils debugUtils_;

    uint32_t graphicsFamilyIndex_ {};
    uint32_t presentFamilyIndex_ {};
    VkQueue graphicsQueue_ {};
    VkQueue presentQueue_ {};
};

} // namespace Base

#endif // BASE_DEVICE_HPP
