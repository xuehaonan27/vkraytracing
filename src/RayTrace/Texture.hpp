#ifndef RAYTRACE_TEXTURE_HPP
#define RAYTRACE_TEXTURE_HPP

#include <memory>
#include <string>

#include "Base/Sampler.hpp"

namespace RayTrace {
class Texture final {
  public:
    static Texture
    LoadTexture(const std::string& filename, const Base::SamplerConfig& samplerConfig);

    Texture& operator=(const Texture&) = delete;
    Texture& operator=(Texture&&) = delete;

    Texture() = default;
    Texture(const Texture&) = default;
    Texture(Texture&&) = default;
    ~Texture() = default;

    const unsigned char* Pixels() const {
        return pixels_.get();
    }

    int Width() const {
        return width_;
    }

    int Height() const {
        return height_;
    }

  private:
    Texture(int width, int height, int channels, unsigned char* pixels);

    Base::SamplerConfig samplerConfig_;
    int width_;
    int height_;
    int channels_;
    std::unique_ptr<unsigned char, void (*)(void*)> pixels_;
};

} // namespace RayTrace

#endif // RAYTRACE_TEXTURE_HPP
