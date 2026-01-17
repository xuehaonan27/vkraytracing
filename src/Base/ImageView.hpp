#ifndef BASE_IMAGE_VIEW_HPP
#define BASE_IMAGE_VIEW_HPP

#include "Utils/Macros.hpp"

namespace Base {
class Device;

class ImageView final {
  public:
    VULKAN_NON_COPIABLE(ImageView)

    explicit ImageView(
        const Device& device,
        VkImage image,
        VkFormat format,
        VkImageAspectFlags aspectFlags
    );
    ~ImageView();

    const class Device& Device() const {
        return device_;
    }

  private:
    const class Device& device_;
    const VkImage image_;
    const VkFormat format_;

    VULKAN_HANDLE(VkImageView, imageView_)
};

} // namespace Base

#endif // BASE_IMAGE_VIEW_HPP
