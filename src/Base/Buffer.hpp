#ifndef BASE_BUFFER_HPP
#define BASE_BUFFER_HPP

#include "DeviceMemory.hpp"
#include "Utils/Macros.hpp"

namespace Base {
class CommandPool;
class Device;

class Buffer final {
  public:
    VULKAN_NON_COPIABLE(Buffer)

    Buffer(const Device& device, size_t size, VkBufferUsageFlags usage);
    ~Buffer();

    const class Device& Device() const {
        return device_;
    }

    DeviceMemory AllocateMemory(VkMemoryPropertyFlags propertyFlags);
    DeviceMemory
    AllocateMemory(VkMemoryAllocateFlags allocateFlags, VkMemoryPropertyFlags propertyFlags);
    VkMemoryRequirements GetMemoryRequirements() const;
    VkDeviceAddress GetDeviceAddress() const;

    void CopyFrom(CommandPool& commandPool, const Buffer& src, VkDeviceSize size);

  private:
    const class Device& device_;

    VULKAN_HANDLE(VkBuffer, buffer_)
};
} // namespace Base

#endif // BASE_BUFFER_HPP
