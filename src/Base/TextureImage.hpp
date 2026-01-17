#ifndef BASE_TEXTURE_IMAGE_HPP
#define BASE_TEXTURE_IMAGE_HPP

#include <memory>
#include "RayTrace/Texture.hpp"
#include "Base/Image.hpp"
#include "Base/ImageView.hpp"
#include "Base/Sampler.hpp"

namespace Base {
class CommandPool;
class DeviceMemory;
class Image;
class ImageView;
class Sampler;

class TextureImage final {
  public:
    TextureImage(const TextureImage&) = delete;
    TextureImage(TextureImage&&) = delete;
    TextureImage& operator=(const TextureImage&) = delete;
    TextureImage& operator=(TextureImage&&) = delete;

    TextureImage(Base::CommandPool& commandPool, const RayTrace::Texture& texture);
    ~TextureImage();

    const Base::ImageView& ImageView() const {
        return *imageView_;
    }

    const Base::Sampler& Sampler() const {
        return *sampler_;
    }

  private:
    std::unique_ptr<Base::Image> image_;
    std::unique_ptr<Base::DeviceMemory> imageMemory_;
    std::unique_ptr<Base::ImageView> imageView_;
    std::unique_ptr<Base::Sampler> sampler_;
};

} // namespace Base

#endif // BASE_TEXTURE_IMAGE_HPP
