#ifndef BASE_UNIFORM_BUFFER_HPP
#define BASE_UNIFORM_BUFFER_HPP

#include <memory>

#include "Buffer.hpp"
#include "Utils/Glm.hpp"

namespace Base {
class Buffer;
class Device;
class DeviceMemory;

class UniformBufferObject {
  public:
    glm::mat4 ModelView;
    glm::mat4 Projection;
    glm::mat4 ModelViewInverse;
    glm::mat4 ProjectionInverse;
    float Aperture;
    float FocusDistance;
    float HeatmapScale;
    uint32_t TotalNumberOfSamples;
    uint32_t NumberOfSamples;
    uint32_t NumberOfBounces;
    uint32_t RandomSeed;
    uint32_t HasSky; // bool
    uint32_t ShowHeatmap; // bool
};

class UniformBuffer {
  public:
    UniformBuffer(const UniformBuffer&) = delete;
    UniformBuffer& operator=(const UniformBuffer&) = delete;
    UniformBuffer& operator=(UniformBuffer&&) = delete;

    explicit UniformBuffer(const Device& device);
    UniformBuffer(UniformBuffer&& other) noexcept;
    ~UniformBuffer();

    const Buffer& Buffer() const {
        return *buffer_;
    }

    void SetValue(const UniformBufferObject& ubo);

  private:
    std::unique_ptr<class Buffer> buffer_;
    std::unique_ptr<DeviceMemory> memory_;
};

} // namespace Base

#endif // BASE_UNIFORM_BUFFER_HPP
