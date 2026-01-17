#ifndef BASE_COMMAND_POOL_HPP
#define BASE_COMMAND_POOL_HPP

#include "Utils/Macros.hpp"

namespace Base {
class Device;

class CommandPool final {
  public:
    VULKAN_NON_COPIABLE(CommandPool)

    CommandPool(const Device& device, uint32_t queueFamilyIndex, bool allowReset);
    ~CommandPool();

    const class Device& Device() const {
        return device_;
    }

  private:
    const class Device& device_;

    VULKAN_HANDLE(VkCommandPool, commandPool_)
};

} // namespace Base

#endif // BASE_COMMAND_POOL_HPP
