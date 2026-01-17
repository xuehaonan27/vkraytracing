#ifndef BASE_SCENE_HPP
#define BASE_SCENE_HPP

#include <memory>
#include <vector>

#include "RayTrace/Texture.hpp"
#include "RayTrace/Model.hpp"
#include "Utils/Macros.hpp"

namespace Base {
class Buffer;
class CommandPool;
class DeviceMemory;
class Image;
class TextureImage;

class Scene final {
  public:
    Scene(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene& operator=(Scene&&) = delete;

    Scene(
        CommandPool& commandPool,
        std::vector<RayTrace::Model>&& models,
        std::vector<RayTrace::Texture>&& textures
    );
    ~Scene();

    const std::vector<RayTrace::Model>& Models() const {
        return models_;
    }

    bool HasProcedurals() const {
        return static_cast<bool>(proceduralBuffer_);
    }

    const Buffer& VertexBuffer() const {
        return *vertexBuffer_;
    }

    const Buffer& IndexBuffer() const {
        return *indexBuffer_;
    }

    const Buffer& MaterialBuffer() const {
        return *materialBuffer_;
    }

    const Buffer& OffsetsBuffer() const {
        return *offsetBuffer_;
    }

    const Buffer& AabbBuffer() const {
        return *aabbBuffer_;
    }

    const Buffer& ProceduralBuffer() const {
        return *proceduralBuffer_;
    }

    const std::vector<VkImageView> TextureImageViews() const {
        return textureImageViewHandles_;
    }

    const std::vector<VkSampler> TextureSamplers() const {
        return textureSamplerHandles_;
    }

  private:
    const std::vector<RayTrace::Model> models_;
    const std::vector<RayTrace::Texture> textures_;

    std::unique_ptr<Buffer> vertexBuffer_;
    std::unique_ptr<DeviceMemory> vertexBufferMemory_;

    std::unique_ptr<Buffer> indexBuffer_;
    std::unique_ptr<DeviceMemory> indexBufferMemory_;

    std::unique_ptr<Buffer> materialBuffer_;
    std::unique_ptr<DeviceMemory> materialBufferMemory_;

    std::unique_ptr<Buffer> offsetBuffer_;
    std::unique_ptr<DeviceMemory> offsetBufferMemory_;

    std::unique_ptr<Buffer> aabbBuffer_;
    std::unique_ptr<DeviceMemory> aabbBufferMemory_;

    std::unique_ptr<Buffer> proceduralBuffer_;
    std::unique_ptr<DeviceMemory> proceduralBufferMemory_;

    std::vector<std::unique_ptr<TextureImage>> textureImages_;
    std::vector<VkImageView> textureImageViewHandles_;
    std::vector<VkSampler> textureSamplerHandles_;
};

} // namespace Base

#endif // BASE_SCENE_HPP
