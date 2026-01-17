#ifndef BASE_DESCRIPTOR_SET_MANAGER_HPP
#define BASE_DESCRIPTOR_SET_MANAGER_HPP

#include <memory>
#include <vector>

#include "DescriptorBinding.hpp"

namespace Base {
class Device;
class DescriptorPool;
class DescriptorSetLayout;
class DescriptorSets;

class DescriptorSetManager final {
  public:
    VULKAN_NON_COPIABLE(DescriptorSetManager)

    explicit DescriptorSetManager(
        const Device& device,
        const std::vector<DescriptorBinding>& descriptorBindings,
        size_t maxSets
    );
    ~DescriptorSetManager();

    const class DescriptorSetLayout& DescriptorSetLayout() const {
        return *descriptorSetLayout_;
    }

    class DescriptorSets& DescriptorSets() {
        return *descriptorSets_;
    }

  private:
    std::unique_ptr<DescriptorPool> descriptorPool_;
    std::unique_ptr<class DescriptorSetLayout> descriptorSetLayout_;
    std::unique_ptr<class DescriptorSets> descriptorSets_;
};

} // namespace Base

#endif // BASE_DESCRIPTOR_SET_MANAGER_HPP
