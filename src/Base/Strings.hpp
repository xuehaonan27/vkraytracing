#ifndef BASE_STRINGS_HPP
#define BASE_STRINGS_HPP

#include "Utils/Macros.hpp"

namespace Base {

class Strings final {
  public:
    VULKAN_NON_COPIABLE(Strings)

    Strings() = delete;
    ~Strings() = delete;

    static const char* DeviceType(VkPhysicalDeviceType deviceType);
    static const char* VendorId(uint32_t vendorId);
};

} // namespace Base

#endif // BASE_STRINGS_HPP