#ifndef BASE_PIPELINE_LAYOUT_HPP
#define BASE_PIPELINE_LAYOUT_HPP

#include "Utils/Macros.hpp"

namespace Base {
class DescriptorSetLayout;
class Device;

class PipelineLayout final {
  public:
    VULKAN_NON_COPIABLE(PipelineLayout)

    PipelineLayout(const Device& device, const DescriptorSetLayout& descriptorSetLayout);
    ~PipelineLayout();

  private:
    const Device& device_;

    VULKAN_HANDLE(VkPipelineLayout, pipelineLayout_)
};

} // namespace Base

#endif // BASE_PIPELINE_LAYOUT_HPP
