#ifndef BASE_DESCRIPTOR_SET_LAYOUT_HPP
#define BASE_DESCRIPTOR_SET_LAYOUT_HPP

#include <vector>

#include "DescriptorBinding.hpp"

namespace Base {
class Device;

class DescriptorSetLayout final {
  public:
    VULKAN_NON_COPIABLE(DescriptorSetLayout)

    DescriptorSetLayout(
        const Device& device,
        const std::vector<DescriptorBinding>& descriptorBindings
    );
    ~DescriptorSetLayout();

  private:
    const Device& device_;

    VULKAN_HANDLE(VkDescriptorSetLayout, layout_)
};

} // namespace Base

#endif // BASE_DESCRIPTOR_SET_LAYOUT_HPP
