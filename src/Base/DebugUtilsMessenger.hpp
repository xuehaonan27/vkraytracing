#ifndef BASE_DEBUG_UTILS_MESSENGER_HPP
#define BASE_DEBUG_UTILS_MESSENGER_HPP

#include "Utils/Macros.hpp"

namespace Base {
class Instance;

class DebugUtilsMessenger final {
  public:
    VULKAN_NON_COPIABLE(DebugUtilsMessenger)

    DebugUtilsMessenger(const Instance& instance, VkDebugUtilsMessageSeverityFlagBitsEXT threshold);
    ~DebugUtilsMessenger();

    VkDebugUtilsMessageSeverityFlagBitsEXT Threshold() const {
        return threshold_;
    }

  private:
    const Instance& instance_;
    const VkDebugUtilsMessageSeverityFlagBitsEXT threshold_;

    VULKAN_HANDLE(VkDebugUtilsMessengerEXT, messenger_)
};

} // namespace Base

#endif // BASE_DEBUG_UTILS_MESSENGER_HPP
